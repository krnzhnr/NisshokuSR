// Demux через libavformat + bitstream-фильтр в Annex-B.
// NVDEC-парсер (cuvid) понимает только Annex-B start codes (00 00 00 01),
// а в MP4/MKV лежат AVCC-пакеты с length-префиксом — без конверсии кадров 0.
#include "Demuxer.h"
#include <cmath>
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
}

struct Demuxer::Impl {
  AVFormatContext* fmt = nullptr;
  AVBSFContext* bsf = nullptr;
  int vstream = -1;
  AVCodecParameters* par = nullptr;
  std::vector<Packet> pending;
  int64_t start_us = 0;
  int64_t video_start_us = 0;
  int64_t total_frames = 0;
};

static int64_t StreamStartUs(AVFormatContext* fmt, int idx) {
  AVStream* st = fmt->streams[idx];
  if (st->start_time == AV_NOPTS_VALUE) return AV_NOPTS_VALUE;
  return av_rescale_q(st->start_time, st->time_base, AV_TIME_BASE_Q);
}

bool Demuxer::open(const std::string& path, int* out_w, int* out_h,
                   std::string* out_codec, double* out_fps,
                   int* out_fps_num, int* out_fps_den) {
  close();
  impl_ = new Impl();
  AVDictionary* opts = nullptr;
  // Большой буфер IO + async: меньше syscalls на HDD/NAS под Windows.
  av_dict_set(&opts, "buffer_size", "4194304", 0);
  if (avformat_open_input(&impl_->fmt, path.c_str(), nullptr, &opts) < 0) return false;
  av_dict_free(&opts);
  if (avformat_find_stream_info(impl_->fmt, nullptr) < 0) return false;
  for (unsigned i = 0; i < impl_->fmt->nb_streams; ++i) {
    if (impl_->fmt->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      impl_->vstream = (int)i;
      impl_->par = impl_->fmt->streams[i]->codecpar;
      break;
    }
  }
  if (impl_->vstream < 0) return false;
  impl_->video_start_us = StreamStartUs(impl_->fmt, impl_->vstream);
  // База — самый ранний поток: сдвиг исходника на неё не делает таймстемпы
  // отрицательными и сохраняет исходное взаимное положение аудио и видео.
  int64_t min_us = AV_NOPTS_VALUE;
  for (unsigned i = 0; i < impl_->fmt->nb_streams; ++i) {
    int64_t us = StreamStartUs(impl_->fmt, (int)i);
    if (us != AV_NOPTS_VALUE && (min_us == AV_NOPTS_VALUE || us < min_us)) min_us = us;
  }
  if (min_us == AV_NOPTS_VALUE) {
    min_us = impl_->fmt->start_time != AV_NOPTS_VALUE ? impl_->fmt->start_time : 0;
  }
  impl_->start_us = min_us;
  if (impl_->video_start_us == AV_NOPTS_VALUE) impl_->video_start_us = min_us;
  // BSF в Annex-B для AVCC-контейнеров (mp4/mov). MKV обычно уже Annex-B.
  const char* bsf_name = nullptr;
  if (impl_->par->codec_id == AV_CODEC_ID_H264) bsf_name = "h264_mp4toannexb";
  else if (impl_->par->codec_id == AV_CODEC_ID_HEVC) bsf_name = "hevc_mp4toannexb";
  if (bsf_name) {
    const AVBitStreamFilter* f = av_bsf_get_by_name(bsf_name);
    if (f && av_bsf_alloc(f, &impl_->bsf) == 0) {
      avcodec_parameters_copy(impl_->bsf->par_in, impl_->par);
      impl_->bsf->time_base_in = impl_->fmt->streams[impl_->vstream]->time_base;
      if (av_bsf_init(impl_->bsf) < 0) {
        av_bsf_free(&impl_->bsf);
        impl_->bsf = nullptr;
      }
    }
  }
  if (out_w) *out_w = impl_->par->width;
  if (out_h) *out_h = impl_->par->height;
  if (out_codec) {
    const AVCodec* c = avcodec_find_decoder(impl_->par->codec_id);
    *out_codec = c ? c->name : "unknown";
  }
  if (out_fps || out_fps_num || out_fps_den) {
    // Точный rational важен: 24000/1001 нельзя округлять до 23976/1000,
    // иначе MP4 получает VFR из-за 3753/3754 тиков в 90 кГц.
    AVRational r = impl_->fmt->streams[impl_->vstream]->avg_frame_rate;
    if (r.num <= 0 || r.den <= 0)
      r = impl_->fmt->streams[impl_->vstream]->r_frame_rate;
    if (r.num <= 0 || r.den <= 0) r = AVRational{30, 1};
    if (out_fps) *out_fps = (double)r.num / r.den;
    if (out_fps_num) *out_fps_num = r.num;
    if (out_fps_den) *out_fps_den = r.den;

    AVStream* vs = impl_->fmt->streams[impl_->vstream];
    if (vs->nb_frames > 0) {
      impl_->total_frames = vs->nb_frames;
    } else {
      double dur = 0.0;
      if (vs->duration != AV_NOPTS_VALUE && vs->time_base.den > 0)
        dur = (double)vs->duration * vs->time_base.num / vs->time_base.den;
      if (dur <= 0 && impl_->fmt->duration != AV_NOPTS_VALUE)
        dur = (double)impl_->fmt->duration / AV_TIME_BASE;
      if (dur > 0) impl_->total_frames = (int64_t)std::llround(dur * r.num / r.den);
    }
  }
  return true;
}

static Packet MkPacket(AVPacket* p) {
  Packet pkt;
  pkt.data.assign(p->data, p->data + p->size);
  pkt.pts = p->pts;
  pkt.key = (p->flags & AV_PKT_FLAG_KEY) != 0;
  return pkt;
}

int64_t Demuxer::startTimeUs() const {
  return impl_ ? impl_->start_us : 0;
}

int64_t Demuxer::videoStartTimeUs() const {
  return impl_ ? impl_->video_start_us : 0;
}

int64_t Demuxer::estimatedFrames() const {
  return impl_ ? impl_->total_frames : 0;
}

bool Demuxer::readNext(Packet& pkt) {
  if (!impl_) return false;
  if (!impl_->pending.empty()) {
    pkt = std::move(impl_->pending.front());
    impl_->pending.erase(impl_->pending.begin());
    return true;
  }
  AVPacket* p = av_packet_alloc();
  bool ok = false;
  while (av_read_frame(impl_->fmt, p) >= 0) {
    if (p->stream_index != impl_->vstream) { av_packet_unref(p); continue; }
    if (impl_->bsf) {
      // Контракт BSF: при успехе фильтр забирает владение и сам сбрасывает
      // пакет — unref делать НЕЛЬЗЯ (двойное освобождение). При ошибке пакет
      // не тронут — тогда unref наш.
      if (av_bsf_send_packet(impl_->bsf, p) < 0) { av_packet_unref(p); continue; }
      AVPacket* f = av_packet_alloc();
      while (av_bsf_receive_packet(impl_->bsf, f) == 0) {
        impl_->pending.push_back(MkPacket(f));
        av_packet_unref(f);
      }
      av_packet_free(&f);
      // p уже сброшен фильтром (владение ушло) — руками не трогаем.
      if (!impl_->pending.empty()) {
        pkt = std::move(impl_->pending.front());
        impl_->pending.erase(impl_->pending.begin());
        ok = true;
        break;
      }
      continue; // BSF съел пакет (SPS/PPS) — читаем дальше
    }
    pkt = MkPacket(p);
    av_packet_unref(p);
    ok = true;
    break;
  }
  av_packet_free(&p);
  return ok;
}

void Demuxer::close() {
  if (impl_) {
    if (impl_->bsf) av_bsf_free(&impl_->bsf);
    if (impl_->fmt) avformat_close_input(&impl_->fmt);
    delete impl_;
    impl_ = nullptr;
  }
}
