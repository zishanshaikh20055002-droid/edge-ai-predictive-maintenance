param(
    [string]$Distro = "Ubuntu",
    [switch]$RunWslMultimodalEval,
    [switch]$ForceSavedModelMode,
    [int]$EvalMaxSamples = 64,
    [int]$EvalRuntimeSamples = 8,
    [int]$EvalBatchSize = 32
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "Missing Python environment: $pythonExe"
}

Write-Host "[verify] compileall app.py + src"
& $pythonExe -m compileall app.py src
if ($LASTEXITCODE -ne 0) {
    throw "compileall failed"
}

Write-Host "[verify] runtime import smoke (app + mqtt_subscriber)"
if ($ForceSavedModelMode) {
    $savedModelPath = Join-Path $repoRoot "models\multimodal_savedmodel"
    if (-not (Test-Path (Join-Path $savedModelPath "saved_model.pb"))) {
        throw "SavedModel artifact not found at $savedModelPath"
    }
    $env:RUNTIME_MODEL_MODE = "multimodal_savedmodel"
    $env:RUNTIME_MODEL_PATH = $savedModelPath
    Write-Host "[verify] forcing runtime mode: multimodal_savedmodel"
}

& $pythonExe -c "import app; print('runtime_mode=', app.runtime_bundle.get('mode')); import src.mqtt_subscriber; print('mqtt_subscriber_import=ok')"
if ($LASTEXITCODE -ne 0) {
    throw "runtime import smoke failed"
}

if ($ForceSavedModelMode) {
    Remove-Item Env:RUNTIME_MODEL_MODE -ErrorAction SilentlyContinue
    Remove-Item Env:RUNTIME_MODEL_PATH -ErrorAction SilentlyContinue
}

Write-Host "[verify] pytest"
$pytestLog = New-TemporaryFile
& $pythonExe -m pytest -q *> $pytestLog
$pytestCode = $LASTEXITCODE
$pytestText = Get-Content $pytestLog -Raw
Get-Content $pytestLog
Remove-Item $pytestLog -Force

if ($pytestCode -ne 0 -and $pytestText -notmatch "no tests ran") {
    throw "pytest failed"
}

if ($pytestCode -ne 0 -and $pytestText -match "no tests ran") {
    Write-Host "[verify] pytest reported no collected tests (treated as informational)"
}

if ($RunWslMultimodalEval) {
    if (-not (Get-Command wsl -ErrorAction SilentlyContinue)) {
        throw "WSL is not available on this system."
    }

    $driveLetter = $repoRoot.Substring(0, 1).ToLowerInvariant()
    $pathSuffix = ($repoRoot.Substring(2) -replace "\\", "/")
    $repoRootWsl = "/mnt/$driveLetter$pathSuffix"

    $bashCommand = "cd '$repoRootWsl'; bash scripts/evaluate_multimodal_mtl_wsl.sh --max-samples $EvalMaxSamples --runtime-samples $EvalRuntimeSamples --batch-size $EvalBatchSize"
    Write-Host "[verify] WSL multimodal evaluation in distro '$Distro'"
    wsl -d $Distro -e bash -lc $bashCommand
    if ($LASTEXITCODE -ne 0) {
        throw "WSL multimodal evaluation failed"
    }
}

Write-Host "[verify] completed"
