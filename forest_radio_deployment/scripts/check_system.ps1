<#
.SYNOPSIS
    Check system resources for forest radio deployment project.
.DESCRIPTION
    Reports GPU, CPU, RAM, disk, Python, CUDA, and Git versions.
    Output saved to outputs/logs/system_information.txt
#>
param(
    [string]$OutFile = "$env:USERPROFILE\forest_radio_deployment\outputs\logs\system_information.txt"
)

$sb = [System.Text.StringBuilder]::new()
$sb.AppendLine("=== System Information ===") | Out-Null
$sb.AppendLine("Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss UTC')") | Out-Null
$sb.AppendLine("") | Out-Null

$sb.AppendLine("=== GPU ===") | Out-Null
try { $gpu = & nvidia-smi 2>&1; $sb.AppendLine($gpu) | Out-Null }
catch { $sb.AppendLine("No NVIDIA GPU detected (nvidia-smi not available)") | Out-Null }
$sb.AppendLine("") | Out-Null

$sb.AppendLine("=== CPU ===") | Out-Null
$cpu = Get-CimInstance Win32_Processor
$sb.AppendLine("Name: $($cpu.Name)") | Out-Null
$sb.AppendLine("Cores: $($cpu.NumberOfCores)") | Out-Null
$sb.AppendLine("Logical Processors: $($cpu.NumberOfLogicalProcessors)") | Out-Null
$sb.AppendLine("") | Out-Null

$sb.AppendLine("=== Memory ===") | Out-Null
$mem = Get-CimInstance Win32_ComputerSystem
$sb.AppendLine("Total RAM: $([math]::Round($mem.TotalPhysicalMemory / 1GB, 1)) GB") | Out-Null
$sb.AppendLine("") | Out-Null

$sb.AppendLine("=== Disk ===") | Out-Null
Get-PSDrive -PSProvider FileSystem | ForEach-Object {
    $usedGB = [math]::Round($_.Used / 1GB, 1)
    $freeGB = [math]::Round($_.Free / 1GB, 1)
    $sb.AppendLine("Drive $($_.Name): Used=$usedGB GB, Free=$freeGB GB") | Out-Null
}
$sb.AppendLine("") | Out-Null

$sb.AppendLine("=== Python ===") | Out-Null
$sb.AppendLine((& python --version 2>&1)) | Out-Null
$sb.AppendLine("") | Out-Null

$sb.AppendLine("=== Git ===") | Out-Null
$sb.AppendLine((& git --version 2>&1)) | Out-Null
$sb.AppendLine("") | Out-Null

$sb.AppendLine("=== OS ===") | Out-Null
$os = Get-CimInstance Win32_OperatingSystem
$sb.AppendLine("$($os.Caption) $($os.Version)") | Out-Null

$content = $sb.ToString()
New-Item -ItemType Directory -Path (Split-Path $OutFile) -Force | Out-Null
$content | Out-File -FilePath $OutFile -Encoding utf8
Write-Host $content
