#Requires -Version 5.1
# Замер прогона nisshoku: per-core CPU + все счётчики GPU (SM/MEM/ENC/DEC/JPG/OFA).
# Запуск:
#   powershell -ExecutionPolicy Bypass -File scripts/bench.ps1 `
#     -In testdata\bench_src.mkv -Engine testdata\animejanai_fp16.engine `
#     -Out testdata\bench.h265 -Label bench -ExtraArgs '--profile quality'
param(
  [Parameter(Mandatory=$true)][string]$In,
  [Parameter(Mandatory=$true)][string]$Engine,
  [Parameter(Mandatory=$true)][string]$Out,
  [string]$Label = "bench",
  [string]$OutDir = "testdata",
  [string]$ExtraArgs = ""
)
$ErrorActionPreference = "Stop"
$OutputEncoding = [Console]::OutputEncoding = [Text.Encoding]::UTF8
$root = Split-Path $PSScriptRoot -Parent
$tp = Join-Path (Split-Path $root -Parent) "third_party"
$trt = (Get-Item env:TRT_ROOT -ErrorAction SilentlyContinue).Value
if (!$trt) { $trt = "$tp\TensorRT-10.8" }
$ff = (Get-Item env:FFMPEG_ROOT -ErrorAction SilentlyContinue).Value
if (!$ff) { $ff = "$tp\ffmpeg-shared\ffmpeg-n8.1-latest-win64-lgpl-shared-8.1" }
$cudaBin = Join-Path $env:CUDA_PATH "bin"
$env:PATH = "$trt\lib;$trt\bin;$ff\bin;$cudaBin;$env:PATH"
if (![System.IO.Path]::IsPathRooted($OutDir)) { $OutDir = Join-Path $root $OutDir }
if (![System.IO.Path]::IsPathRooted($In)) { $In = Join-Path $root $In }
if (![System.IO.Path]::IsPathRooted($Engine)) { $Engine = Join-Path $root $Engine }
if (![System.IO.Path]::IsPathRooted($Out)) { $Out = Join-Path $root $Out }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$prefix = Join-Path $OutDir "$Label-$stamp"
$runOut = "$prefix.run.out.log"
$runErr = "$prefix.run.err.log"
$gpuCsv = "$prefix.gpu.csv"
$cpuCsv = "$prefix.cpu.csv"
$procCsv = "$prefix.proc.csv"
$pmonTxt = "$prefix.pmon.txt"
$summaryTxt = "$prefix.summary.txt"

$exe = "$root\build\nisshoku.exe"
if (!(Test-Path $exe)) { throw "no exe: $exe" }
$extraList = @()
if ($ExtraArgs.Trim()) { $extraList = $ExtraArgs.Trim() -split '[\s,]+' }
$allArgs = @("--in", $In, "--engine", $Engine, "--out", $Out) + $extraList

# Start-Process склеивает ArgumentList через пробел и не экранирует пути,
# поэтому собираем строку сами (кириллица/пробелы в путях — обычное дело).
function QuoteArg([string]$s) {
  if ($s -match '[\s"]') { return '"' + $s.Replace('"', '\"') + '"' }
  return $s
}
$argLine = ($allArgs | ForEach-Object { QuoteArg $_ }) -join ' '

# GPU: снимаем все доступные движки каждые 500 мс. nvidia-smi крутится,
# пока не убьем после выхода пайплайна.
$gpuFields = "timestamp,utilization.gpu,utilization.memory,utilization.encoder,utilization.decoder,utilization.jpeg,utilization.ofa,clocks.sm,clocks.mem,power.draw,temperature.gpu,fan.speed,memory.used"
$gpuProc = Start-Process -FilePath "nvidia-smi" `
  -ArgumentList @("--query-gpu=$gpuFields", "--format=csv,noheader,nounits", "-lms", "500") `
  -RedirectStandardOutput $gpuCsv -PassThru -WindowStyle Hidden

$sw = [Diagnostics.Stopwatch]::StartNew()
$proc = Start-Process -FilePath $exe -ArgumentList $argLine `
  -RedirectStandardOutput $runOut -RedirectStandardError $runErr -PassThru -NoNewWindow

$cpuRows = New-Object System.Collections.ArrayList
$procRows = New-Object System.Collections.ArrayList
$lastCpu = $null
$lastCpuT = $null
$lastBg = $null
$prevRaw = @{}
$pmonDone = $false
while (!$proc.HasExited) {
  # per-core из Win32_PerfRawData: считаем дельту idle/timestamp сами
  # (локализованные имена счётчиков Get-Counter на русской Windows не работают).
  try {
    $raw = Get-CimInstance Win32_PerfRawData_PerfOS_Processor -ErrorAction Stop
    if ($prevRaw.Count -gt 0) {
      foreach ($x in $raw) {
        $p = $prevRaw[$x.Name]
        if ($null -eq $p) { continue }
        $dt = [double]$x.Timestamp_Sys100NS - [double]$p.Timestamp_Sys100NS
        $di = [double]$x.PercentIdleTime - [double]$p.PercentIdleTime
        $cpu = 0.0
        if ($dt -gt 0) { $cpu = 100.0 - ($di / $dt * 100.0) }
        [void]$cpuRows.Add([pscustomobject]@{
          t = [math]::Round($sw.Elapsed.TotalSeconds, 1)
          core = $x.Name
          pct = [math]::Round($cpu, 2)
        })
      }
    }
    $prevRaw = @{}
    foreach ($x in $raw) { $prevRaw[$x.Name] = $x }
  } catch { }
  $proc.Refresh()
  $cpuSec = $proc.TotalProcessorTime.TotalSeconds
  if ($null -ne $lastCpu) {
    $dt = $sw.Elapsed.TotalSeconds - $lastCpuT
    if ($dt -gt 0) {
      $bgCpu = 0.0
      $ff = @(Get-Process ffmpeg -ErrorAction SilentlyContinue)
      foreach ($f in $ff) { $bgCpu += $f.CPU }
      $bgDelta = if ($null -ne $lastBg) { $bgCpu - $lastBg } else { 0 }
      [void]$procRows.Add([pscustomobject]@{
        t = [math]::Round($sw.Elapsed.TotalSeconds, 2)
        cores = [math]::Round(($cpuSec - $lastCpu) / $dt, 2) # во сколько ядер
        bg_cores = [math]::Round($bgDelta / $dt, 2)
        rss_mb = [math]::Round($proc.WorkingSet64 / 1MB, 0)
      })
      $lastBg = $bgCpu
    }
  }
  if ($null -eq $lastBg) {
    $lastBg = 0.0
    foreach ($f in @(Get-Process ffmpeg -ErrorAction SilentlyContinue)) { $lastBg += $f.CPU }
  }
  $lastCpu = $cpuSec
  $lastCpuT = $sw.Elapsed.TotalSeconds
  if (!$pmonDone -and $sw.Elapsed.TotalSeconds -ge 8) {
    # разовая выборка по процессам: sm/mem/enc/dec
    & nvidia-smi pmon -s um -c 1 2>&1 | Out-File -FilePath $pmonTxt -Encoding utf8
    $pmonDone = $true
  }
  Start-Sleep -Milliseconds 300
}
$proc.WaitForExit()
$wall = $sw.Elapsed.TotalSeconds
Start-Sleep -Milliseconds 1200
if ($gpuProc -and !$gpuProc.HasExited) { $gpuProc.Kill() }
$gpuProc.WaitForExit()
if ($cpuRows.Count) { $cpuRows | Export-Csv -Path $cpuCsv -NoTypeInformation -Encoding UTF8 }
if ($procRows.Count) { $procRows | Export-Csv -Path $procCsv -NoTypeInformation -Encoding UTF8 }

function Stats($values) {
  $v = @($values | Where-Object { $_ -ne $null -and "$_" -ne "" -and "$_" -notmatch 'N/A' } | ForEach-Object { [double]$_ })
  if ($v.Count -eq 0) { return $null }
  $m = $v | Measure-Object -Average -Minimum -Maximum
  return [pscustomobject]@{
    avg = [math]::Round($m.Average, 1)
    min = [math]::Round($m.Minimum, 1)
    max = [math]::Round($m.Maximum, 1)
    n = $v.Count
  }
}

# --- GPU CSV -> метрики ---
$gpu = @()
if (Test-Path $gpuCsv) {
  foreach ($line in Get-Content $gpuCsv) {
    if (!$line.Trim()) { continue }
    $p = $line -split ','
    if ($p.Count -lt 13) { continue }
    $gpu += [pscustomobject]@{
      ts = $p[0].Trim()
      sm = $p[1].Trim(); mem = $p[2].Trim(); enc = $p[3].Trim(); dec = $p[4].Trim()
      jpg = $p[5].Trim(); ofa = $p[6].Trim()
      clk_sm = $p[7].Trim(); clk_mem = $p[8].Trim()
      pwr = $p[9].Trim(); temp = $p[10].Trim(); fan = $p[11].Trim(); vram = $p[12].Trim()
    }
  }
}
$gpuStats = [ordered]@{}
if ($gpu.Count) {
  foreach ($k in 'sm','mem','enc','dec','jpg','ofa','clk_sm','clk_mem','pwr','temp','fan','vram') {
    $gpuStats[$k] = Stats ($gpu | ForEach-Object { $_.$k })
  }
}

# --- CPU: средняя/макс по каждому ядру и общая ---
$cores = $cpuRows | Where-Object { $_.core -ne '_Total' } | Group-Object core
$coreStats = foreach ($g in $cores) {
  $st = Stats ($g.Group | ForEach-Object { $_.pct })
  [pscustomobject]@{ core = $g.Name; avg = $st.avg; max = $st.max }
}
$totalRow = $cpuRows | Where-Object { $_.core -eq '_Total' }
$totalStats = Stats ($totalRow | ForEach-Object { $_.pct })
$procStats = Stats ($procRows | ForEach-Object { $_.cores })
$bgStats = Stats ($procRows | ForEach-Object { $_.bg_cores })
$rssStats = Stats ($procRows | ForEach-Object { $_.rss_mb })

$outFile = Get-Item $Out -ErrorAction SilentlyContinue
$frames = 0
$procSec = 0.0
$fpsProc = 0.0
if (Test-Path $runOut) {
  $m = Select-String -Path $runOut -Pattern 'done, frames=(\d+), ([0-9.]+) s, ([0-9.]+) fps' | Select-Object -Last 1
  if ($m) {
    $frames = [int]$m.Matches[0].Groups[1].Value
    $procSec = [double]$m.Matches[0].Groups[2].Value
    $fpsProc = [double]$m.Matches[0].Groups[3].Value
  }
}
$fps = if ($wall -gt 0 -and $frames -gt 0) { [math]::Round($frames / $wall, 2) } else { 0 }

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("=== $Label ===")
$lines.Add("cmd: nisshoku.exe $argLine")
$lines.Add("wall_s=$([math]::Round($wall,2))  frames=$frames  fps_e2e=$fps  proc_s=$procSec  fps_proc=$fpsProc")
$lines.Add("cpu_total_avg=$($totalStats.avg)%  cpu_total_max=$($totalStats.max)%  proc_cores_avg=$($procStats.avg)  bg_ffmpeg_cores_avg=$($bgStats.avg)  rss_avg_mb=$($rssStats.avg)")
$lines.Add("cpu_per_core (avg/max %):")
foreach ($c in ($coreStats | Sort-Object { [int]$_.core })) {
  $lines.Add(("  core {0,2}: {1,6} / {2,6}" -f $c.core, $c.avg, $c.max))
}
$lines.Add("gpu:")
foreach ($k in $gpuStats.Keys) {
  $s = $gpuStats[$k]
  if ($s) { $lines.Add(("  {0,-7} avg={1,7} min={2,7} max={3,7} n={4}" -f $k, $s.avg, $s.min, $s.max, $s.n)) }
}
if ($outFile) { $lines.Add("out_bytes=$($outFile.Length)") }
$lines | Set-Content -Path $summaryTxt -Encoding utf8
$lines | ForEach-Object { Write-Host $_ }
Write-Host "`nlogs: $prefix.*"
