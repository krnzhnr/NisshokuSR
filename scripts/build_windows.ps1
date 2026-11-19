#Requires -Version 5.1
# Build via Ninja (no CUDA .props needed) + portable cmake/ninja from third_party.
# CUDA 12.8 + VS18/2026 host compiler is unsupported by nvcc (cudafe crashes),
# so the script picks a VS2022 cl (14.3x/14.4x) and loads its env via vcvars64.bat.
param([string]$Config = "Release")
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$tp = Join-Path (Split-Path $root -Parent) "third_party"
foreach ($v in "NVCODEC_SDK_DIR", "TRT_ROOT", "FFMPEG_ROOT") {
  if (-not (Get-Item env:$v -ErrorAction SilentlyContinue).Value) { throw "Missing env $v (setx + new terminal)" }
}
$cmake = "$tp\cmake-portable\cmake-4.5.0-windows-x86_64\bin\cmake.exe"
$ninja = "$tp\ninja\ninja.exe"
if (!(Test-Path $cmake)) { throw "No portable cmake: $cmake" }
if (!(Test-Path $ninja)) { throw "No portable ninja: $ninja" }

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$installs = & $vswhere -products * -property installationPath 2>$null
$cls = foreach ($vs in $installs) {
  Get-ChildItem "$vs\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe" -ErrorAction SilentlyContinue
}
if (!$cls) { throw "cl.exe not found. Check Desktop C++ workload in VS Installer" }
$cl22 = $cls | Where-Object { $_.FullName -match "\\14\.(3\d|4\d)" } | Sort-Object FullName -Descending | Select-Object -First 1
if ($cl22) { $cl = $cl22 }
else {
  throw "Only VS18/2026 compiler found. nvcc 12.8 crashes on it. Install VS2022 Build Tools (Desktop C++) side by side."
}
$vcroot = $cl.DirectoryName
for ($k = 0; $k -lt 6; $k++) { $vcroot = Split-Path $vcroot }  # ...\VC
$aux = Join-Path $vcroot "Auxiliary\Build\vcvars64.bat"
if (!(Test-Path $aux)) { throw "vcvars64.bat not found near $($cl.FullName)" }
$toolset = ($cl.DirectoryName -split '[\\/]')[-4]
$ver = $toolset -replace "^(\d+\.\d+).*", '$1'
$vcout = cmd /c ('"' + $aux + '" -vcvars_ver=' + $ver + ' >NUL & set')
foreach ($line in $vcout) {
  $i = $line.IndexOf("=")
  if ($i -gt 0) { Set-Item "env:$($line.Substring(0, $i))" $line.Substring($i + 1) }
}
$cudaBin = Join-Path $env:CUDA_PATH "bin"
if (Test-Path $cudaBin) { $env:PATH = "$cudaBin;$env:PATH" }

& $cmake -S $root -B "$root\build" -G Ninja "-DCMAKE_MAKE_PROGRAM=$ninja" "-DCMAKE_BUILD_TYPE=$Config"
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }
& $cmake --build "$root\build" --config $Config --parallel
if ($LASTEXITCODE -ne 0) { throw "build failed" }
Write-Host "OK: $root\build\nisshoku.exe"
