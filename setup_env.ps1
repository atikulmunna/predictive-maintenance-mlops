# Predictive Maintenance MLOps Platform Setup Script
# Run this script to create conda environment and install dependencies

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Predictive Maintenance MLOps Setup" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$ENV_NAME = "pred-maint"
$PYTHON_VERSION = "3.11"

# Check if conda is available
Write-Host "[1/6] Checking conda installation..." -ForegroundColor Yellow
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Conda not found! Please install Anaconda or Miniconda first." -ForegroundColor Red
    Write-Host "Download from: https://docs.conda.io/en/latest/miniconda.html" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Conda found: $(conda --version)" -ForegroundColor Green
Write-Host ""

# Check if environment already exists
Write-Host "[2/6] Checking if environment '$ENV_NAME' exists..." -ForegroundColor Yellow
$env_exists = conda env list | Select-String -Pattern $ENV_NAME -Quiet
if ($env_exists) {
    Write-Host "⚠️  Environment '$ENV_NAME' already exists!" -ForegroundColor Yellow
    $response = Read-Host "Do you want to remove and recreate it? (y/N)"
    if ($response -eq 'y' -or $response -eq 'Y') {
        Write-Host "Removing existing environment..." -ForegroundColor Yellow
        conda env remove -n $ENV_NAME -y
        Write-Host "✅ Existing environment removed" -ForegroundColor Green
    } else {
        Write-Host "❌ Setup cancelled. Use existing environment or choose different name." -ForegroundColor Red
        exit 1
    }
}
Write-Host ""

# Create conda environment
Write-Host "[3/6] Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..." -ForegroundColor Yellow
conda create -n $ENV_NAME python=$PYTHON_VERSION -y
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to create conda environment" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Conda environment created successfully" -ForegroundColor Green
Write-Host ""

# Install PyTorch with CUDA support
Write-Host "[4/6] Installing PyTorch with CUDA 11.8 support..." -ForegroundColor Yellow
Write-Host "This may take 5-10 minutes..." -ForegroundColor Gray
conda install -n $ENV_NAME pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install PyTorch" -ForegroundColor Red
    exit 1
}
Write-Host "✅ PyTorch with CUDA installed successfully" -ForegroundColor Green
Write-Host ""

# Activate environment and install requirements
Write-Host "[5/6] Installing project dependencies..." -ForegroundColor Yellow
Write-Host "Installing base requirements..." -ForegroundColor Gray
conda run -n $ENV_NAME pip install -r requirements/base.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install base requirements" -ForegroundColor Red
    exit 1
}

Write-Host "Installing training requirements..." -ForegroundColor Gray
conda run -n $ENV_NAME pip install -r requirements/training.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install training requirements" -ForegroundColor Red
    exit 1
}

Write-Host "Installing serving requirements..." -ForegroundColor Gray
conda run -n $ENV_NAME pip install -r requirements/serving.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install serving requirements" -ForegroundColor Red
    exit 1
}

Write-Host "Installing development requirements..." -ForegroundColor Gray
conda run -n $ENV_NAME pip install -r requirements/dev.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install dev requirements" -ForegroundColor Red
    exit 1
}
Write-Host "✅ All dependencies installed successfully" -ForegroundColor Green
Write-Host ""

# Setup environment file
Write-Host "[6/6] Setting up environment configuration..." -ForegroundColor Yellow
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "✅ Created .env file from .env.example" -ForegroundColor Green
    Write-Host "⚠️  Please edit .env and update configuration values!" -ForegroundColor Yellow
} else {
    Write-Host "⚠️  .env file already exists, skipping..." -ForegroundColor Yellow
}
Write-Host ""

# Create .gitkeep files in data directories
New-Item -ItemType File -Path "data/raw/.gitkeep" -Force | Out-Null
New-Item -ItemType File -Path "data/processed/.gitkeep" -Force | Out-Null
New-Item -ItemType File -Path "data/features/.gitkeep" -Force | Out-Null
New-Item -ItemType File -Path "data/models/.gitkeep" -Force | Out-Null

# Verify GPU availability
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Verifying GPU Setup" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Running GPU check..." -ForegroundColor Yellow
conda run -n $ENV_NAME python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
Write-Host ""

# Success message
Write-Host "================================================" -ForegroundColor Green
Write-Host "  ✅ Setup Complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Activate environment:" -ForegroundColor White
Write-Host "   conda activate $ENV_NAME" -ForegroundColor Yellow
Write-Host ""
Write-Host "2. Edit .env file with your configuration" -ForegroundColor White
Write-Host ""
Write-Host "3. Start exploring the dataset:" -ForegroundColor White
Write-Host "   jupyter lab" -ForegroundColor Yellow
Write-Host "   # Create notebooks/01_eda.ipynb" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Check GPU is working:" -ForegroundColor White
Write-Host "   python -c 'import torch; print(torch.cuda.is_available())'" -ForegroundColor Yellow
Write-Host ""
Write-Host "Happy coding! 🚀" -ForegroundColor Green
