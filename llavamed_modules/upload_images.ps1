# PowerShell script to compress and upload images to Lambda
# Usage: .\upload_images.ps1 -LambdaIP "your-lambda-ip"

param(
    [Parameter(Mandatory=$true)]
    [string]$LambdaIP,
    
    [Parameter(Mandatory=$false)]
    [string]$LambdaUser = "ubuntu",
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipCompress
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Uploading Images to Lambda" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

$imagesDir = "C:\Users\PC\Downloads\extracted_archive\patches_captions\patches_captions"
$zipFile = "C:\Users\PC\Downloads\images.zip"

# Verify images directory exists
if (-not (Test-Path $imagesDir)) {
    Write-Host "❌ ERROR: Images directory not found: $imagesDir" -ForegroundColor Red
    exit 1
}

# Count files
$fileCount = (Get-ChildItem $imagesDir -File | Measure-Object).Count
Write-Host "Found $fileCount image files" -ForegroundColor Yellow
Write-Host ""

if (-not $SkipCompress) {
    Write-Host "Step 1: Compressing images..." -ForegroundColor Yellow
    Write-Host "⚠️  This will take 10-30 minutes and create a large zip file" -ForegroundColor Red
    Write-Host ""
    
    if (Test-Path $zipFile) {
        $overwrite = Read-Host "Zip file already exists. Overwrite? (yes/no)"
        if ($overwrite -ne "yes") {
            Write-Host "Using existing zip file: $zipFile" -ForegroundColor Yellow
        } else {
            Remove-Item $zipFile -Force
            Write-Host "Compressing images (this will take a while)..." -ForegroundColor Yellow
            Compress-Archive -Path "$imagesDir\*" -DestinationPath $zipFile -CompressionLevel Optimal
        }
    } else {
        Write-Host "Compressing images (this will take a while)..." -ForegroundColor Yellow
        Compress-Archive -Path "$imagesDir\*" -DestinationPath $zipFile -CompressionLevel Optimal
    }
    
    $zipSize = (Get-Item $zipFile).Length / 1GB
    Write-Host "✅ Compression complete. Zip file size: $([math]::Round($zipSize, 2)) GB" -ForegroundColor Green
    Write-Host ""
    
    Write-Host "Step 2: Uploading zip file..." -ForegroundColor Yellow
    Write-Host "⚠️  This will also take some time depending on your connection speed" -ForegroundColor Red
    Write-Host ""
    
    scp $zipFile "${LambdaUser}@${LambdaIP}:/home/ubuntu/data/images.zip"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to upload zip file" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "✅ Zip file uploaded" -ForegroundColor Green
    Write-Host ""
    
    Write-Host "Step 3: Extract on Lambda..." -ForegroundColor Yellow
    Write-Host "SSH into Lambda and run:" -ForegroundColor Cyan
    Write-Host "  ssh ${LambdaUser}@${LambdaIP}" -ForegroundColor White
    Write-Host "  cd /home/ubuntu/data" -ForegroundColor White
    Write-Host "  unzip images.zip -d histopathology_dataset/images/" -ForegroundColor White
    Write-Host "  rm images.zip  # Clean up" -ForegroundColor White
    Write-Host ""
    
} else {
    Write-Host "Skipping compression, uploading directly..." -ForegroundColor Yellow
    Write-Host "⚠️  WARNING: This will upload ~262k individual files and take HOURS!" -ForegroundColor Red
    Write-Host ""
    $confirm = Read-Host "Continue with direct upload? (yes/no)"
    if ($confirm -ne "yes") {
        Write-Host "Cancelled" -ForegroundColor Yellow
        exit 0
    }
    
    Write-Host "Uploading images (this will take a very long time)..." -ForegroundColor Yellow
    scp -r "${imagesDir}\*" "${LambdaUser}@${LambdaIP}:/home/ubuntu/data/histopathology_dataset/images/"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to upload images" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "✅ Images uploaded" -ForegroundColor Green
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Upload Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""


