# Direct upload commands - copy and paste these into PowerShell
# Replace YOUR_LAMBDA_IP with your actual Lambda IP address

$LAMBDA_IP = "YOUR_LAMBDA_IP"  # <-- CHANGE THIS
$LAMBDA_USER = "ubuntu"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Uploading LLaVA-Med Integration to Lambda" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Verify paths
$modulesDir = "C:\Users\PC\Downloads\llavamed_modules"
$datasetCSV = "C:\Users\PC\Downloads\patchgastricadc222_prepared\dataset_with_labels.csv"
$imagesDir = "C:\Users\PC\Downloads\extracted_archive\patches_captions\patches_captions"

if (-not (Test-Path $modulesDir)) {
    Write-Host "❌ ERROR: Modules directory not found!" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $datasetCSV)) {
    Write-Host "❌ ERROR: Dataset CSV not found!" -ForegroundColor Red
    exit 1
}

Write-Host "Step 1: Uploading modules directory..." -ForegroundColor Yellow
scp -r "$modulesDir" "${LAMBDA_USER}@${LAMBDA_IP}:/home/ubuntu/projects/llavamed-integration/"
Write-Host "✅ Modules uploaded" -ForegroundColor Green
Write-Host ""

Write-Host "Step 2: Creating remote directories..." -ForegroundColor Yellow
ssh "${LAMBDA_USER}@${LAMBDA_IP}" "mkdir -p /home/ubuntu/data/histopathology_dataset/images /home/ubuntu/data/histopathology_outputs"
Write-Host "✅ Directories created" -ForegroundColor Green
Write-Host ""

Write-Host "Step 3: Uploading dataset CSV..." -ForegroundColor Yellow
scp "$datasetCSV" "${LAMBDA_USER}@${LAMBDA_IP}:/home/ubuntu/data/histopathology_dataset/dataset_with_labels.csv"
Write-Host "✅ Dataset CSV uploaded" -ForegroundColor Green
Write-Host ""

Write-Host "Step 4: Image upload..." -ForegroundColor Yellow
Write-Host "⚠️  WARNING: Uploading ~262k images will take a VERY long time!" -ForegroundColor Red
Write-Host "   You can skip this and upload images later if needed." -ForegroundColor Yellow
Write-Host ""
$confirm = Read-Host "Upload images now? (yes/no)"
if ($confirm -eq "yes") {
    Write-Host "Uploading images (this may take hours)..." -ForegroundColor Yellow
    scp -r "${imagesDir}\*" "${LAMBDA_USER}@${LAMBDA_IP}:/home/ubuntu/data/histopathology_dataset/images/"
    Write-Host "✅ Images uploaded" -ForegroundColor Green
} else {
    Write-Host "⏭️  Skipped image upload. Upload later with:" -ForegroundColor Yellow
    Write-Host "   scp -r `"$imagesDir\*`" ${LAMBDA_USER}@${LAMBDA_IP}:/home/ubuntu/data/histopathology_dataset/images/" -ForegroundColor White
}
Write-Host ""

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Upload Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next: SSH into Lambda and run setup:" -ForegroundColor Yellow
Write-Host "  ssh ${LAMBDA_USER}@${LAMBDA_IP}" -ForegroundColor White
Write-Host "  cd /home/ubuntu/projects/llavamed-integration" -ForegroundColor White
Write-Host "  chmod +x llavamed_modules/lambda_setup.sh" -ForegroundColor White
Write-Host "  ./llavamed_modules/lambda_setup.sh" -ForegroundColor White
Write-Host ""


