# PowerShell script to upload files to Lambda Labs
# Usage: .\upload_to_lambda.ps1 -LambdaIP "your-lambda-ip"

param(
    [Parameter(Mandatory=$true)]
    [string]$LambdaIP,
    
    [Parameter(Mandatory=$false)]
    [string]$LambdaUser = "ubuntu"
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Uploading LLaVA-Med Integration to Lambda" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if SCP is available (usually via OpenSSH on Windows 10+)
$scpAvailable = Get-Command scp -ErrorAction SilentlyContinue
if (-not $scpAvailable) {
    Write-Host "❌ ERROR: SCP not found. Please install OpenSSH client:" -ForegroundColor Red
    Write-Host "   Settings > Apps > Optional Features > OpenSSH Client" -ForegroundColor Yellow
    exit 1
}

$baseDir = "C:\Users\PC\Downloads"
$modulesDir = Join-Path $baseDir "llavamed_modules"
$datasetCSV = Join-Path $baseDir "patchgastricadc222_prepared\dataset_with_labels.csv"
$imagesDir = Join-Path $baseDir "extracted_archive\patches_captions\patches_captions"

# Verify files exist
if (-not (Test-Path $modulesDir)) {
    Write-Host "❌ ERROR: Modules directory not found: $modulesDir" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $datasetCSV)) {
    Write-Host "❌ ERROR: Dataset CSV not found: $datasetCSV" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $imagesDir)) {
    Write-Host "❌ ERROR: Images directory not found: $imagesDir" -ForegroundColor Red
    exit 1
}

Write-Host "Step 1: Uploading modules directory..." -ForegroundColor Yellow
$remoteModulesPath = "/home/ubuntu/projects/llavamed-integration"
scp -r "$modulesDir" "${LambdaUser}@${LambdaIP}:$remoteModulesPath/"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to upload modules" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Modules uploaded" -ForegroundColor Green
Write-Host ""

Write-Host "Step 2: Creating remote directories..." -ForegroundColor Yellow
ssh "${LambdaUser}@${LambdaIP}" "mkdir -p /home/ubuntu/data/histopathology_dataset/images /home/ubuntu/data/histopathology_outputs"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to create directories" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Directories created" -ForegroundColor Green
Write-Host ""

Write-Host "Step 3: Uploading dataset CSV..." -ForegroundColor Yellow
scp "$datasetCSV" "${LambdaUser}@${LambdaIP}:/home/ubuntu/data/histopathology_dataset/dataset_with_labels.csv"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to upload CSV" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Dataset CSV uploaded" -ForegroundColor Green
Write-Host ""

Write-Host "Step 4: Uploading images..." -ForegroundColor Yellow
Write-Host "⚠️  WARNING: This will upload ~262k image files. This may take a VERY long time!" -ForegroundColor Red
Write-Host "   Consider compressing the images first or using rsync for better performance." -ForegroundColor Yellow
Write-Host ""
$confirm = Read-Host "Continue with image upload? (yes/no)"
if ($confirm -ne "yes") {
    Write-Host "Skipping image upload. You can upload images later manually." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To upload images later, run:" -ForegroundColor Cyan
    Write-Host "  scp -r `"$imagesDir\*`" ${LambdaUser}@${LambdaIP}:/home/ubuntu/data/histopathology_dataset/images/" -ForegroundColor White
    exit 0
}

# Upload images (this will be slow)
scp -r "$imagesDir\*" "${LambdaUser}@${LambdaIP}:/home/ubuntu/data/histopathology_dataset/images/"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to upload images" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Images uploaded" -ForegroundColor Green
Write-Host ""

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Upload Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. SSH into Lambda: ssh ${LambdaUser}@${LambdaIP}" -ForegroundColor White
Write-Host "2. Run setup: cd /home/ubuntu/projects/llavamed-integration && ./llavamed_modules/lambda_setup.sh" -ForegroundColor White
Write-Host "3. Start processing: python3 llavamed_modules/run.py" -ForegroundColor White
Write-Host ""


