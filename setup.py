"""
Setup script for MCP News Analysis Project
Helps install dependencies and configure the environment
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible. Need Python 3.8+")
        return False

def create_virtual_environment():
    """Create and activate virtual environment"""
    print("\n📦 Setting up virtual environment...")
    
    if not os.path.exists(".venv"):
        if run_command("python -m venv .venv", "Creating virtual environment"):
            print("✅ Virtual environment created at .venv")
        else:
            return False
    else:
        print("✅ Virtual environment already exists")
    
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📋 Installing dependencies...")
    
    # Get the pip path based on OS
    if os.name == 'nt':  # Windows
        pip_path = ".venv\\Scripts\\pip"
    else:  # Unix/Linux/MacOS
        pip_path = ".venv/bin/pip"
    
    # Upgrade pip first
    run_command(f"{pip_path} install --upgrade pip", "Upgrading pip")
    
    # Install core requirements
    if run_command(f"{pip_path} install -r requirements.txt", "Installing core Python packages"):
        print("✅ Core Python packages installed successfully")
    else:
        print("❌ Failed to install core packages. Try installing manually:")
        print("   pip install mcp fastmcp httpx python-dotenv mistralai langchain langchain-mistralai")
        return False
    
    # Ask about development dependencies
    print("\n🛠️ Optional: Install Development Dependencies?")
    install_dev = input("Install development tools? (y/n): ").lower().strip()
    
    if install_dev in ['y', 'yes']:
        if run_command(f"{pip_path} install -r requirements-dev.txt", "Installing development packages"):
            print("✅ Development packages installed successfully")
        else:
            print("❌ Failed to install development packages")
    
    # Install NLTK data
    if os.name == 'nt':  # Windows
        python_path = ".venv\\Scripts\\python"
    else:  # Unix/Linux/MacOS
        python_path = ".venv/bin/python"
    
    nltk_command = f'{python_path} -c "import nltk; nltk.download(\'vader_lexicon\'); nltk.download(\'punkt\'); nltk.download(\'stopwords\')"'
    run_command(nltk_command, "Installing NLTK data")
    
    return True

def check_config():
    """Check configuration file"""
    print("\n⚙️ Checking configuration...")
    
    config_file = Path("config/.env")
    if config_file.exists():
        print("✅ Configuration file exists")
        
        # Read and check for API keys
        with open(config_file, 'r') as f:
            content = f.read()
            
        if "your_mistral_api_key_here" in content:
            print("⚠️  Please set your Mistral API key in config/.env")
            print("   Get your key from: https://console.mistral.ai/")
        else:
            print("✅ Mistral API key appears to be set")
            
        if "RAPIDAPI_KEY=" in content:
            print("✅ RapidAPI key is configured")
        else:
            print("❌ RapidAPI key not found in config")
            
    else:
        print("❌ Configuration file not found")
        return False
    
    return True

def run_tests():
    """Run basic tests to verify setup"""
    print("\n🧪 Running basic tests...")
    
    if os.name == 'nt':  # Windows
        python_path = ".venv\\Scripts\\python"
    else:  # Unix/Linux/MacOS
        python_path = ".venv/bin/python"
    
    if run_command(f"{python_path} test_tools.py", "Running component tests"):
        print("✅ Basic tests completed")
        return True
    else:
        print("⚠️  Some tests failed - check the output above")
        return False

def print_next_steps():
    """Print instructions for next steps"""
    print("\n" + "="*60)
    print("🎉 Setup completed!")
    print("="*60)
    
    print("\n📋 Available Run Options:")
    print("1.  Terminal Interface:")
    print("   .\\run_agent.bat     # Windows")  
    print("   ./run_agent.sh      # Linux/Mac")
    
    print("\n2. 🔧 Manual Setup:")
    if os.name == 'nt':  # Windows
        print("   .venv\\Scripts\\Activate")
    else:  # Unix/Linux/MacOS
        print("   source .venv/bin/activate")
    
    print("\n⚙️ Configuration:")
    print("1. Set your Mistral API key in config/.env")
    print("   Get your key from: https://console.mistral.ai/")
    print("2. Set your RapidAPI key for news fetching")
    print("   Get your key from: https://rapidapi.com/")
    
    print("\n🧪 Testing:")
    print("   python test_tools.py              # Test individual tools")
    print("   python client/mcp_client.py       # Test enhanced client")
    
    print("\n📚 Available Requirements Files:")
    print("   requirements.txt        # Core dependencies")
    print("   requirements-dev.txt    # Development tools")
    
    print("\n🔗 Access Points:")
    print("   Terminal Interface: Interactive CLI")
    print("   MCP Server: Runs automatically as needed")

def main():
    """Main setup function"""
    print("🚀 MCP News Analysis Project Setup")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Create virtual environment
    if not create_virtual_environment():
        return
    
    # Install dependencies
    if not install_dependencies():
        return
    
    # Check configuration
    check_config()
    
    # Run tests (optional)
    print("\n🧪 Run basic tests?")
    run_tests_choice = input("Run component tests? (y/n): ").lower().strip()
    if run_tests_choice in ['y', 'yes']:
        run_tests()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()
