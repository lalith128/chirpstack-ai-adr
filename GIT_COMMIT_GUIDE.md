# Git Commit Guide for chirpstack-ai-adr

## Initial Setup

```bash
cd /mnt/c/Users/LALITH\ V/Desktop/doc/chirpstack-docker

# Initialize git (if not already done)
git init

# Add remote
git remote add origin https://github.com/lalith128/chirpstack-ai-adr.git
```

## Files to Commit

### Core Files
- `README.md` - Main documentation
- `LICENSE` - MIT License
- `.gitignore` - Git ignore rules
- `docker-compose.yml` - Docker Compose configuration

### ADR Engine
- `adr_engine/adr_engine_phase1.py` - Main ADR engine
- `adr_engine/requirements.txt` - Python dependencies
- `adr_engine/Dockerfile` - Docker image

### Codec
- `codec/codec_bme280.js` - JavaScript decoder for ChirpStack

### Examples
- `examples/arduino_bme280_node/arduino_bme280_node.ino` - Arduino example

### Documentation
- `docs/QUICKSTART.md` - Quick start guide
- `docs/SETUP.md` - Detailed setup instructions
- `docs/ARCHITECTURE.md` - Architecture documentation
- `docs/ADR_DISABLED_CONFIRMATION.md` - ADR disable confirmation

### Models
- `models/lora_env_shield_float32.tflite` - TFLite model
- `models/scaler_p0_context.pkl` - Context scaler
- `models/scaler_p0_env.pkl` - Environment scaler
- `models/scaler_p0_target.pkl` - Target scaler
- `models/encoder_p0_action.pkl` - Action encoder

### Data Directory
- `data/.gitkeep` - Keep data directory in git

## Commit Commands

```bash
# Stage all files
git add README.md LICENSE docker-compose.yml
git add adr_engine/
git add codec/
git add examples/
git add docs/
git add models/
git add data/.gitkeep

# Check status
git status

# Commit
git commit -m "Initial commit: Phase 1 AI-Driven LoRaWAN Data Collector

- Implemented external ADR engine with AI Safety Shield
- Added TensorFlow Lite model for SNR prediction
- Created exploration strategy with weighted random selection
- Implemented CSV logging for RL training data
- Added comprehensive documentation and examples
- Configured Docker deployment with docker-compose
- Disabled native ADR in all ChirpStack region files
- Included BME280 sensor decoder and Arduino example"

# Push to GitHub
git branch -M main
git push -u origin main
```

## Subsequent Updates

```bash
# Add changed files
git add .

# Commit with descriptive message
git commit -m "Update: Brief description of changes"

# Push
git push
```

## Branch Strategy (Optional)

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "Feature: Description"

# Push feature branch
git push -u origin feature/your-feature-name

# Create pull request on GitHub
# After merge, switch back to main
git checkout main
git pull
```

## Files NOT to Commit (Already in .gitignore)

- `data/*.csv` - CSV data files (except .gitkeep)
- `configuration/postgresql/data/` - PostgreSQL data
- `configuration/redis/data/` - Redis data
- `__pycache__/` - Python cache
- `.vscode/`, `.idea/` - IDE files
- `*.log` - Log files

## Verification

After pushing, verify on GitHub:
1. Visit https://github.com/lalith128/chirpstack-ai-adr
2. Check all files are present
3. Verify README.md renders correctly
4. Test clone on another machine
