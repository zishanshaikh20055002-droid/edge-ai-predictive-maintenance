param(
    [string]$Distro = "Ubuntu",
    [string]$ModelPath = "models/best_multimodal_mtl.keras",
    [string]$OutDir = "models/multimodal_savedmodel"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command wsl -ErrorAction SilentlyContinue)) {
    throw "WSL is not available on this system."
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$driveLetter = $repoRoot.Substring(0, 1).ToLowerInvariant()
$pathSuffix = ($repoRoot.Substring(2) -replace "\\", "/")
$repoRootWsl = "/mnt/$driveLetter$pathSuffix"

$bashCommand = "cd '$repoRootWsl'; bash scripts/export_savedmodel_multimodal_wsl.sh --model '$ModelPath' --out '$OutDir'"

Write-Host "Exporting multimodal SavedModel in WSL distro '$Distro'"
wsl -d $Distro -e bash -lc $bashCommand
