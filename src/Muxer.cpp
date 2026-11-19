#include "Muxer.h"
#include <windows.h>
#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/pixfmt.h>
}

namespace {

std::wstring Utf8ToWide(const std::string& s) {
  if (s.empty()) return {};
  int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
  if (n <= 1) return {};
  std::wstring w((size_t)(n - 1), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
  return w;
}

std::string WideToUtf8(const std::wstring& w) {
  if (w.empty()) return {};
  int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, nullptr, 0, nullptr, nullptr);
  if (n <= 1) return {};
  std::string s((size_t)(n - 1), '\0');
  WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, s.data(), n, nullptr, nullptr);
  return s;
}

std::wstring Quote(const std::wstring& s) {
  return L"\"" + s + L"\"";
}

bool EndsWithNoCase(const std::string& s, const char* suffix) {
  const size_t n = std::strlen(suffix);
  if (s.size() < n) return false;
  for (size_t i = 0; i < n; ++i) {
    unsigned char a = (unsigned char)s[s.size() - n + i];
    unsigned char b = (unsigned char)suffix[i];
    if (std::tolower(a) != std::tolower(b)) return false;
  }
  return true;
}

std::string AvError(int code) {
  char buf[AV_ERROR_MAX_STRING_SIZE] = {};
  av_strerror(code, buf, sizeof(buf));
  return buf;
}

std::wstring FindFfmpeg() {
  wchar_t exe_buf[32767] = {};
  DWORD n = GetModuleFileNameW(nullptr, exe_buf, (DWORD)std::size(exe_buf));
  if (n == 0 || n >= std::size(exe_buf)) return {};
  std::filesystem::path exe_dir = std::filesystem::path(exe_buf).parent_path();

  std::vector<std::filesystem::path> candidates;
  candidates.push_back(exe_dir / L"ffmpeg.exe");

  wchar_t env[32767] = {};
  DWORD en = GetEnvironmentVariableW(L"FFMPEG_ROOT", env, (DWORD)std::size(env));
  if (en > 0 && en < std::size(env)) {
    candidates.push_back(std::filesystem::path(env) / L"bin" / L"ffmpeg.exe");
  }

  // dev-раскладка: <Utils>\third_party\ffmpeg-shared\<build>\bin\ffmpeg.exe
  std::filesystem::path utils = exe_dir.parent_path().parent_path();
  std::filesystem::path ffroot = utils / L"third_party" / L"ffmpeg-shared";
  std::error_code ec;
  if (std::filesystem::is_directory(ffroot, ec)) {
    for (const auto& entry : std::filesystem::directory_iterator(ffroot, ec)) {
      if (!entry.is_directory()) continue;
      candidates.push_back(entry.path() / L"bin" / L"ffmpeg.exe");
    }
  }

  for (const auto& p : candidates) {
    if (std::filesystem::is_regular_file(p, ec)) return p.wstring();
  }

  wchar_t found[32767] = {};
  if (SearchPathW(nullptr, L"ffmpeg.exe", nullptr, (DWORD)std::size(found), found, nullptr) > 0)
    return found;
  return {};
}

std::string ReadFileTail(const std::wstring& path, size_t max_bytes = 12000) {
  std::ifstream in(std::filesystem::path(path), std::ios::binary);
  if (!in) return {};
  std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (data.size() > max_bytes) data = data.substr(data.size() - max_bytes);
  return data;
}

bool RunProcess(const std::wstring& exe, const std::wstring& args,
                const std::wstring& log_path, std::string& error) {
  SECURITY_ATTRIBUTES sa = {};
  sa.nLength = sizeof(sa);
  sa.bInheritHandle = TRUE;
  HANDLE log = CreateFileW(log_path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, &sa,
                           CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (log == INVALID_HANDLE_VALUE) {
    error = "не удалось открыть лог FFmpeg";
    return false;
  }

  STARTUPINFOW si = {};
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESTDHANDLES;
  si.hStdOutput = log;
  si.hStdError = log;
  si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);

  PROCESS_INFORMATION pi = {};
  std::wstring cmd = Quote(exe) + L" " + args;
  std::vector<wchar_t> cmdline(cmd.begin(), cmd.end());
  cmdline.push_back(0);

  BOOL ok = CreateProcessW(exe.c_str(), cmdline.data(), nullptr, nullptr, TRUE,
                           CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
  CloseHandle(log);
  if (!ok) {
    error = "не удалось запустить ffmpeg (error " + std::to_string(GetLastError()) + ")";
    return false;
  }

  WaitForSingleObject(pi.hProcess, INFINITE);
  DWORD code = 1;
  GetExitCodeProcess(pi.hProcess, &code);
  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);

  if (code != 0) {
    std::string log_text = ReadFileTail(log_path);
    error = "ffmpeg remux завершился с кодом " + std::to_string(code);
    if (!log_text.empty()) error += ": " + log_text;
    error += " (лог: " + WideToUtf8(log_path) + ")";
    return false;
  }

  DeleteFileW(log_path.c_str());
  return true;
}

bool WritePacket(AVFormatContext* fmt, AVStream* st, const EncPacket& p,
                 int64_t pts, int64_t dts, AVRational src_tb, std::string& error) {
  if (p.data.size() > (size_t)INT_MAX) {
    error = "пакет больше INT_MAX";
    return false;
  }
  AVPacket* pkt = av_packet_alloc();
  if (!pkt) {
    error = "av_packet_alloc failed";
    return false;
  }
  int ret = av_new_packet(pkt, (int)p.data.size());
  if (ret < 0) {
    av_packet_free(&pkt);
    error = "av_new_packet: " + AvError(ret);
    return false;
  }
  memcpy(pkt->data, p.data.data(), p.data.size());
  pkt->pts = pts;
  pkt->dts = dts;
  pkt->duration = 1;
  pkt->stream_index = st->index;
  pkt->flags = p.key ? AV_PKT_FLAG_KEY : 0;
  av_packet_rescale_ts(pkt, src_tb, st->time_base);
  ret = av_interleaved_write_frame(fmt, pkt);
  av_packet_free(&pkt); // функция забирает данные и оставляет пакет пустым
  if (ret < 0) {
    error = "av_interleaved_write_frame: " + AvError(ret);
    return false;
  }
  return true;
}

} // namespace

struct Muxer::Impl {
  AVFormatContext* fmt = nullptr;
  AVStream* st = nullptr;
  std::string ts_path;
  std::string out_path;
  std::string source_path;
  bool copy_audio = true;
  bool copy_subs = true;
  bool trim_to_video = false;
  int64_t dts = 0;
  int64_t written = 0;
  int64_t shift_ticks = 0;
  int delay_frames = 0;
  int fps_num = 24000, fps_den = 1001;
  AVRational src_tb{1000, 24000};
  int64_t source_start_us = 0;
  int64_t source_video_start_us = 0;
  bool started = false;
  std::vector<EncPacket> pending;
  bool opened = false;
  bool finished = false;
  bool success = false;
  bool keep_temp = false;
};

Muxer::~Muxer() {
  if (impl_) {
    if (!impl_->finished) abort();
    delete impl_;
    impl_ = nullptr;
  }
}

bool Muxer::open(const MuxConfig& cfg, std::string& error) {
  abort();
  impl_ = new Impl();
  impl_->out_path = cfg.out_path;
  impl_->source_path = cfg.source_path;
  impl_->copy_audio = cfg.copy_audio;
  impl_->copy_subs = cfg.copy_subs;
  impl_->trim_to_video = cfg.trim_to_video;
  impl_->fps_num = cfg.fps_num;
  impl_->fps_den = cfg.fps_den;
  impl_->src_tb = AVRational{cfg.fps_den, cfg.fps_num};
  impl_->source_start_us = cfg.source_start_us;
  impl_->source_video_start_us = cfg.source_video_start_us;
  impl_->ts_path = cfg.out_path + ".tmp.ts";

  // MPEG-TS (промежуточный контейнер) не умеет AV1: поток станет bin_data,
  // и remux не сможет его сопоставить. Raw AV1 поддерживается отдельно.
  if (cfg.codec == "av1") {
    error = "AV1 в MKV/MP4 пока не поддерживается (используйте raw .av1)";
    abort();
    return false;
  }

  int ret = avformat_alloc_output_context2(&impl_->fmt, nullptr, "mpegts", impl_->ts_path.c_str());
  if (ret < 0 || !impl_->fmt) {
    error = "avformat_alloc_output_context2: " + AvError(ret);
    abort();
    return false;
  }

  impl_->st = avformat_new_stream(impl_->fmt, nullptr);
  if (!impl_->st) {
    error = "avformat_new_stream failed";
    abort();
    return false;
  }

  AVCodecParameters* par = impl_->st->codecpar;
  par->codec_type = AVMEDIA_TYPE_VIDEO;
  if (cfg.codec == "hevc" || cfg.codec == "h265") par->codec_id = AV_CODEC_ID_HEVC;
  else if (cfg.codec == "av1") par->codec_id = AV_CODEC_ID_AV1;
  else par->codec_id = AV_CODEC_ID_H264;
  par->width = cfg.width;
  par->height = cfg.height;
  par->format = cfg.bit_depth >= 10 ? AV_PIX_FMT_YUV420P10LE : AV_PIX_FMT_YUV420P;
  par->color_range = AVCOL_RANGE_MPEG;
  par->color_primaries = AVCOL_PRI_BT709;
  par->color_trc = AVCOL_TRC_BT709;
  par->color_space = AVCOL_SPC_BT709;
  par->framerate = AVRational{cfg.fps_num, cfg.fps_den};
  if (par->codec_id == AV_CODEC_ID_HEVC)
    par->profile = cfg.bit_depth >= 10 ? AV_PROFILE_HEVC_MAIN_10 : AV_PROFILE_HEVC_MAIN;
  else if (par->codec_id == AV_CODEC_ID_H264)
    par->profile = AV_PROFILE_H264_HIGH;
  else if (par->codec_id == AV_CODEC_ID_AV1)
    par->profile = AV_PROFILE_AV1_MAIN;

  impl_->st->time_base = AVRational{cfg.fps_den, cfg.fps_num};
  impl_->st->avg_frame_rate = AVRational{cfg.fps_num, cfg.fps_den};
  impl_->st->r_frame_rate = AVRational{cfg.fps_num, cfg.fps_den};

  ret = avio_open(&impl_->fmt->pb, impl_->ts_path.c_str(), AVIO_FLAG_WRITE);
  if (ret < 0) {
    error = "avio_open " + impl_->ts_path + ": " + AvError(ret);
    abort();
    return false;
  }
  ret = avformat_write_header(impl_->fmt, nullptr);
  if (ret < 0) {
    error = "avformat_write_header: " + AvError(ret);
    abort();
    return false;
  }
  impl_->opened = true;
  return true;
}

bool Muxer::startPending(std::string& error) {
  if (!impl_ || impl_->started) return true;
  if (!impl_->fmt || !impl_->st) {
    error = "muxer не открыт";
    return false;
  }

  // TS требует PTS >= DTS; сдвигаем PTS на минимальную задержку по первым кадрам.
  int64_t max_deficit = 0;
  for (size_t i = 0; i < impl_->pending.size(); ++i) {
    int64_t deficit = (int64_t)i - impl_->pending[i].pts;
    if (deficit > max_deficit) max_deficit = deficit;
  }
  impl_->delay_frames = (int)(max_deficit + 1);
  // Ровно тот сдвиг, который получает PTS видео в TS (90 кГц). При remux
  // вычитаем его, чтобы вернуть display-время; для MP4 двигаем и DTS, тогда
  // edit list сам уберет задержку переупорядочивания B-кадров.
  if (!impl_->pending.empty()) {
    impl_->shift_ticks = av_rescale_q(impl_->pending.front().pts + impl_->delay_frames,
                                      impl_->src_tb, AVRational{1, 90000});
  }

  for (size_t i = 0; i < impl_->pending.size(); ++i) {
    if (!WritePacket(impl_->fmt, impl_->st, impl_->pending[i],
                     impl_->pending[i].pts + impl_->delay_frames, (int64_t)i,
                     impl_->src_tb, error))
      return false;
  }
  impl_->dts = (int64_t)impl_->pending.size();
  impl_->written += (int64_t)impl_->pending.size();
  impl_->pending.clear();
  impl_->started = true;
  return true;
}

bool Muxer::write(const std::vector<EncPacket>& packets, std::string& error) {
  if (!impl_ || !impl_->fmt || !impl_->st) {
    error = "muxer не открыт";
    return false;
  }

  for (const auto& p : packets) {
    if (!impl_->started) {
      impl_->pending.push_back(p);
      if (impl_->pending.size() >= 64) {
        if (!startPending(error)) return false;
      }
      continue;
    }
    if (!WritePacket(impl_->fmt, impl_->st, p, p.pts + impl_->delay_frames,
                     impl_->dts++, impl_->src_tb, error))
      return false;
    ++impl_->written;
  }
  return true;
}

bool Muxer::finish(std::string& error) {
  if (!impl_) {
    error = "muxer не открыт";
    return false;
  }
  if (!impl_->opened) {
    error = "muxer не открыт";
    return false;
  }
  if (impl_->finished) {
    if (!impl_->success) error = "muxer уже завершён с ошибкой";
    return impl_->success;
  }

  if (!startPending(error)) {
    impl_->keep_temp = true;
    abort();
    return false;
  }
  if (impl_->written == 0) {
    error = "не закодировано ни одного кадра";
    impl_->keep_temp = true;
    abort();
    return false;
  }
  int ret = av_write_trailer(impl_->fmt);
  if (ret < 0) {
    error = "av_write_trailer: " + AvError(ret);
    impl_->keep_temp = true;
    abort();
    return false;
  }
  if (impl_->fmt->pb) avio_closep(&impl_->fmt->pb);
  avformat_free_context(impl_->fmt);
  impl_->fmt = nullptr;
  impl_->st = nullptr;

  std::wstring ffmpeg = FindFfmpeg();
  if (ffmpeg.empty()) {
    error = "ffmpeg.exe не найден (положите рядом с exe или задайте FFMPEG_ROOT)";
    impl_->finished = true;
    impl_->keep_temp = true;
    return false;
  }

  const bool is_mp4 = EndsWithNoCase(impl_->out_path, ".mp4");
  const std::wstring ts = Utf8ToWide(impl_->ts_path);
  const std::wstring src = Utf8ToWide(impl_->source_path);
  const std::wstring out = Utf8ToWide(impl_->out_path);

  // Видео пайплайна начинается с нуля, а исходник может начинаться позже
  // (например, MKV с start_time 0.083). Переносим аудио/субтитры на ту же
  // базу, иначе получаем рассинхрон на величину start_time.
  const int64_t base_us = impl_->source_start_us;
  const int64_t video_base_us =
      std::max<int64_t>(0, impl_->source_video_start_us - base_us);

  std::wstring args = L"-hide_banner -loglevel error -y -copyts ";
  args += L"-i " + Quote(ts);
  if (base_us != 0) {
    std::wostringstream off;
    off << std::fixed << std::setprecision(6)
        << L" -itsoffset " << (-(double)base_us / AV_TIME_BASE);
    args += off.str();
  }
  args += L" -i " + Quote(src);
  args += L" -map 0:v:0";
  if (impl_->copy_audio) args += L" -map 1:a?";
  if (impl_->copy_subs && !is_mp4)
    args += L" -map 1:s? -c:s copy -map 1:t? -c:t copy";
  args += L" -c copy -map_metadata 1";
  // В MP4 ffmpeg пишет главы отдельной дорожкой, и некоторые демуксеры потом
  // ругаются на битую ссылку; для MKV главы нативные и безопасные.
  args += is_mp4 ? L" -map_chapters -1" : L" -map_chapters 1";

  // Сдвиг PTS/DTS видео назад: в TS они были увеличены ради PTS >= DTS.
  // MKV хранит только PTS, но ffmpeg CLI при PTS < DTS «угадывает» DTS и
  // заодно портит PTS, поэтому двигаем обе шкалы; matroska DTS игнорирует.
  const int64_t video_delta_ticks =
      av_rescale_q(video_base_us, AV_TIME_BASE_Q, AVRational{1, 90000}) -
      impl_->shift_ticks;
  if (video_delta_ticks != 0) {
    std::wstring sign = video_delta_ticks > 0 ? L"+" : L"";
    std::wostringstream bsf;
    bsf << L" -bsf:v setts=pts=TS" << sign << video_delta_ticks
        << L":dts=TS" << sign << video_delta_ticks;
    args += bsf.str();
  }

  if (impl_->trim_to_video) {
    // Только для диагностического --frames N: иначе теряется аудио-хвост.
    std::wostringstream dur;
    dur << std::fixed << std::setprecision(6) << L" -t "
        << ((double)video_base_us / AV_TIME_BASE +
            (double)impl_->written * impl_->fps_den / impl_->fps_num);
    args += dur.str();
  }
  // Отрицательный DTS начала — нормальная часть edit list для B-кадров;
  // для MKV DTS не хранится, поэтому его тоже можно отдать как есть.
  args += L" -avoid_negative_ts disabled";
  if (is_mp4) {
    // Точная CFR-сетка: при 24000/1001 каждый sample — 1001 тик, а не
    // смесь 3753/3754 из 90-кГц TS (иначе MediaInfo показывает VFR).
    std::wostringstream scale;
    scale << L" -video_track_timescale " << impl_->fps_num;
    args += scale.str();
    args += L" -use_editlist 1 -movflags +faststart -f mp4";
  } else {
    args += L" -f matroska";
  }
  args += L" " + Quote(out);

  const std::wstring log = ts + L".ffmpeg.log";
  if (!RunProcess(ffmpeg, args, log, error)) {
    error += " (временный TS: " + impl_->ts_path + ")";
    impl_->finished = true;
    impl_->keep_temp = true;
    return false;
  }

  if (std::getenv("NISSHOKU_KEEP_TS")) {
    fprintf(stderr, "[mux] keep temporary: %s\n", impl_->ts_path.c_str());
  } else {
    DeleteFileW(ts.c_str());
  }
  impl_->finished = true;
  impl_->success = true;
  return true;
}

void Muxer::abort() {
  if (!impl_) return;
  if (impl_->fmt) {
    if (impl_->fmt->pb) avio_closep(&impl_->fmt->pb);
    avformat_free_context(impl_->fmt);
    impl_->fmt = nullptr;
    impl_->st = nullptr;
  }
  if (!impl_->keep_temp) {
    std::wstring ts = Utf8ToWide(impl_->ts_path);
    if (!ts.empty()) DeleteFileW(ts.c_str());
    std::wstring log = ts + L".ffmpeg.log";
    if (!ts.empty()) DeleteFileW(log.c_str());
  }
  impl_->finished = true;
  impl_->success = false;
}
