<#
.SYNOPSIS
    Download FOR-instanceV2 dataset from Zenodo, verify MD5 checksums, and extract.

.DESCRIPTION
    Downloads 3 ZIP files from Zenodo record 16742708:
      - clean_forestformer.zip  (pretrained model weights)
      - test_data.zip           (29 test PLY files)
      - train_val_data.zip      (65 train/val PLY files)
    Verifies MD5 checksums and extracts to subdirectories.
    Supports resume via re-running (overwrites partial downloads).

.PARAMETER DATA_DIR
    Destination directory for downloads and extraction.
    Default: $HOME\forest_radio_deployment\data\raw\for_instance_v2
#>
param(
    [string]$DATA_DIR = "$env:USERPROFILE\forest_radio_deployment\data\raw\for_instance_v2"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Path $DATA_DIR -Force | Out-Null
Set-Location $DATA_DIR

Write-Host "=============================="
Write-Host "Disk space before download"
Write-Host "=============================="
$drive = Get-PSDrive C
Write-Host "C: Free=$([math]::Round($drive.Free/1GB,1)) GB"

$files = @(
    @{ Name = "clean_forestformer.zip"; URL = "https://zenodo.org/records/16742708/files/clean_forestformer.zip?download=1"; MD5 = "553d67379331966509076f3fbb409e57" },
    @{ Name = "test_data.zip";          URL = "https://zenodo.org/records/16742708/files/test_data.zip?download=1";          MD5 = "1c00a0f0b89f03b74064432162619136" },
    @{ Name = "train_val_data.zip";     URL = "https://zenodo.org/records/16742708/files/train_val_data.zip?download=1";     MD5 = "5a63cc1cbe88edd9ebec28ad7e46f79b" }
)

# Download
foreach ($f in $files) {
    $outPath = Join-Path $DATA_DIR $f.Name
    Write-Host "`n=============================="
    Write-Host "Downloading $($f.Name)..."
    Write-Host "=============================="
    $retries = 0; $maxRetries = 10; $success = $false
    while (-not $success -and $retries -lt $maxRetries) {
        try {
            $wc = New-Object System.Net.WebClient
            $wc.DownloadFile($f.URL, $outPath)
            $success = $true
            Write-Host "Download complete: $($f.Name)"
        } catch {
            $retries++
            Write-Host "Retry $retries/$maxRetries for $($f.Name): $_"
            Start-Sleep -Seconds 10
        }
    }
    if (-not $success) { Write-Host "FAILED: $($f.Name)"; exit 1 }
    $size = (Get-Item $outPath).Length
    Write-Host "File size: $([math]::Round($size/1MB,1)) MB"
}

# Verify MD5
Write-Host "`n=============================="
Write-Host "Verifying MD5 checksums"
Write-Host "=============================="
$allPassed = $true
foreach ($f in $files) {
    $path = Join-Path $DATA_DIR $f.Name
    $hash = (Get-FileHash -Path $path -Algorithm MD5).Hash.ToLower()
    if ($hash -eq $f.MD5) {
        Write-Host "PASS: $($f.Name) ($hash)"
    } else {
        Write-Host "FAIL: $($f.Name) expected=$($f.MD5) actual=$hash"
        $allPassed = $false
    }
}
if (-not $allPassed) { Write-Host "MD5 verification FAILED"; exit 1 }

# Extract
Write-Host "`n=============================="
Write-Host "Extracting archives"
Write-Host "=============================="
foreach ($name in @("clean_forestformer", "test_data", "train_val_data")) {
    $zip = Join-Path $DATA_DIR "$name.zip"
    $dest = Join-Path $DATA_DIR $name
    New-Item -ItemType Directory -Path $dest -Force | Out-Null
    Write-Host "Extracting $name.zip..."
    Expand-Archive -Path $zip -DestinationPath $dest -Force
}

# Manifests
Write-Host "`n=============================="
Write-Host "Creating manifests"
Write-Host "=============================="
foreach ($name in @("forestformer_code", "test_data", "train_val_data")) {
    $dir = Join-Path $DATA_DIR $name
    if (-not (Test-Path $dir)) { $dir = Join-Path $DATA_DIR ($name -replace '_code','') }
    if (-not (Test-Path $dir)) { continue }
    $manifest = Join-Path $DATA_DIR "${name}_manifest.csv"
    "path,size_bytes" | Out-File $manifest -Encoding utf8
    Get-ChildItem $dir -Recurse -File | ForEach-Object {
        "$($_.FullName.Replace($DATA_DIR + '\', '')),$($_.Length)" | Out-File $manifest -Append -Encoding utf8
    }
    $count = (Get-ChildItem $dir -Recurse -File | Measure-Object).Count
    $sizeGB = [math]::Round((Get-ChildItem $dir -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1GB, 2)
    Write-Host "$name : $count files, $sizeGB GB"
}

Write-Host "`n=============================="
Write-Host "Done. Disk space after:"
Write-Host "=============================="
$drive = Get-PSDrive C
Write-Host "C: Free=$([math]::Round($drive.Free/1GB,1)) GB"
