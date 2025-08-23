import sys
import os
import subprocess

def test_web_script():
    
    if not os.path.exists('venv'):
        print("❌ Virtual environment not found. Please run:")
        print("   python3 -m venv venv")
        print("   source venv/bin/activate")
        print("   pip install beautifulsoup4 requests openai tqdm aiohttp")
        return False
    
    test_url = "https://example.com"
    test_words = "test,example"
    
    print(f"🧪 Testing web.py with URL: {test_url}")
    print(f"🔍 Search words: {test_words}")
    print("=" * 50)
    
    try:
        cmd = [
            sys.executable, "web.py",
            "--url", test_url,
            "--words", test_words,
            "--max-pages", "2",
            "--no-ai"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        print("📤 Command Output:")
        print(result.stdout)
        
        if result.stderr:
            print("⚠️  Errors/Warnings:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ Test completed successfully!")
            return True
        else:
            print(f"❌ Test failed with return code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_web_script()
    sys.exit(0 if success else 1)
