param(
    [string]$Distro = "Ubuntu",
    [int]$Epochs = 40,
    [int]$BatchSize = 32,
    [switch]$RebuildDataset,
    [switch]$NoBootstrap,
    [switch]$ProbeOnly
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command wsl -ErrorAction SilentlyContinue)) {
    throw "WSL is not available on this system."
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$driveLetter = $repoRoot.Substring(0, 1).ToLowerInvariant()
$pathSuffix = ($repoRoot.Substring(2) -replace "\\", "/")
$repoRootWsl = "/mnt/$driveLetter$pathSuffix"

$launcherArgs = @()
if ($ProbeOnly) {
    $launcherArgs += "--probe-only"
} else {
    $launcherArgs += "--epochs"
    $launcherArgs += "$Epochs"
    $launcherArgs += "--batch-size"
    $launcherArgs += "$BatchSize"

    if ($RebuildDataset) {
        $launcherArgs += "--rebuild-dataset"
    }
    if ($NoBootstrap) {
        $launcherArgs += "--no-bootstrap"
    }
}

$quotedArgs = ($launcherArgs | ForEach-Object { "'$_'" }) -join " "
$bashCommand = "cd '" + $repoRootWsl + "'; bash scripts/train_multimodal_mtl_gpu_wsl.sh " + $quotedArgs

Write-Host "Launching WSL GPU training in distro '$Distro' from $repoRootWsl"
wsl -d $Distro -e bash -lc $bashCommand