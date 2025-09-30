#!/usr/bin/env python3
"""
ProfAI WebSocket Server Startup Script
Starts both the FastAPI server and WebSocket server for optimal performance
"""

import asyncio
import threading
import time
import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

  
if hasattr(sys, '_clear_type_cache'):
    sys._clear_type_cache()

# Clear any existing module cache for our services
modules_to_clear = [mod for mod in sys.modules if 'sarvam' in mod.lower()]
for mod in modules_to_clear:
    del sys.modules[mod]

def start_fastapi_server(host, port):
    """Start FastAPI server in a separate thread with proper async handling."""
    try:
        import subprocess
        import sys

        print(f"🚀 Starting FastAPI server with Gunicorn using config file...")

        # Command to run Gunicorn with a configuration file
        # This is the standard production approach.
        command = [
            "gunicorn",
            "-c", "gunicorn_config.py", # Use the configuration file
            "app:app"
        ]

        # Start the Gunicorn process
        process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)
        process.wait()  # Wait for the process to complete
    except Exception as e:
        print(f"❌ FastAPI startup error: {e}")
        import traceback
        traceback.print_exc()

async def start_websocket_server_async(host, port):
    """Start WebSocket server asynchronously."""
    from websocket_server import start_websocket_server
    
    print(f"🌐 Starting WebSocket server on ws://{host}:{port}")
    await start_websocket_server(host, port)

def main():
    """Main startup function."""
    # Get host and port from environment variables
    fastapi_host = os.getenv("HOST", "0.0.0.0")
    fastapi_port = int(os.getenv("PORT", 5001))
    websocket_host = os.getenv("WEBSOCKET_HOST", "0.0.0.0")
    websocket_port = int(os.getenv("WEBSOCKET_PORT", 8765))

    print("=" * 60)
    print("🎓 ProfAI - High Performance WebSocket Server")
    print("=" * 60)
    print(f"FastAPI Server: http://{fastapi_host}:{fastapi_port}")
    print(f"WebSocket Server: ws://{websocket_host}:{websocket_port}")
    print("=" * 60)
    
    try:
        # Check if ports are available
        import socket
        
        def check_port(host, port, name):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                print(f"⚠️ Warning: Port {port} ({name}) appears to be in use")
                return False
            return True
        
        fastapi_available = check_port(fastapi_host, fastapi_port, "FastAPI")
        websocket_available = check_port(websocket_host, websocket_port, "WebSocket")
        
        if not fastapi_available or not websocket_available:
            print("💡 If you're running existing servers, you can:")
            print("   - Stop them and restart this script")
            
            response = input("\nContinue anyway? (y/N): ").lower().strip()
            if response != 'y':
                print("👋 Exiting...")
                return
        
        # Start FastAPI server in background thread
        print("\n🚀 Starting FastAPI server...")
        fastapi_thread = threading.Thread(target=start_fastapi_server, args=(fastapi_host, fastapi_port), daemon=True)
        fastapi_thread.start()
        
        # Give FastAPI time to start
        time.sleep(3)
        
        print("✅ FastAPI server started")
        print(f"📱 Web interface: http://{fastapi_host}:{fastapi_port}")
        print(f"🧪 WebSocket test: http://{fastapi_host}:{fastapi_port}/profai-websocket-test")
        print("\n🌐 Starting WebSocket server...")
        
        # Start WebSocket server (blocking) with proper async handling
        asyncio.run(start_websocket_server_async(websocket_host, websocket_port))
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers...")
        print("👋 Goodbye!")
    except Exception as e:
        print(f"\n💥 Error starting servers: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Check if ports 5001 and 8765 are available")
        print("   2. Verify all dependencies are installed")
        print("   3. Check the error message above for specific issues")
        sys.exit(1)

if __name__ == "__main__":
    main()
