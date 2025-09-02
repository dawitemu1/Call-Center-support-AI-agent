# FFmpeg Installation Script for Windows
# Run this script as Administrator in PowerShell

Write-Host "🔧 FFmpeg Installation Script" -ForegroundColor Green
Write-Host "This script will download and install FFmpeg for your Speech-Text application." -ForegroundColor Yellow

# Create FFmpeg directory
$ffmpegDir = "C:\ffmpeg"
$ffmpegBin = "$ffmpegDir\bin"

try {
    # Check if FFmpeg already exists
    if (Test-Path "$ffmpegBin\ffmpeg.exe") {
        Write-Host "✅ FFmpeg is already installed at $ffmpegBin" -ForegroundColor Green
        Write-Host "Adding to PATH..." -ForegroundColor Yellow
    } else {
        Write-Host "📥 Downloading FFmpeg..." -ForegroundColor Yellow
        
        # Create directory
        New-Item -ItemType Directory -Force -Path $ffmpegDir | Out-Null
        
        # Download FFmpeg (Windows build)
        $downloadUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
        $zipFile = "$ffmpegDir\ffmpeg.zip"
        
        Invoke-WebRequest -Uri $downloadUrl -OutFile $zipFile
        
        Write-Host "📦 Extracting FFmpeg..." -ForegroundColor Yellow
        Expand-Archive -Path $zipFile -DestinationPath $ffmpegDir -Force
        
        # Move files to correct location
        $extractedFolder = Get-ChildItem -Path $ffmpegDir -Directory | Where-Object { $_.Name -like "ffmpeg-*" } | Select-Object -First 1
        if ($extractedFolder) {
            Move-Item "$($extractedFolder.FullName)\bin\*" -Destination $ffmpegBin -Force
            Remove-Item $extractedFolder.FullName -Recurse -Force
        }
        
        # Clean up
        Remove-Item $zipFile -Force
        
        Write-Host "✅ FFmpeg installed successfully!" -ForegroundColor Green
    }
    
    # Add to PATH
    $currentPath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
    if ($currentPath -notlike "*$ffmpegBin*") {
        Write-Host "🔧 Adding FFmpeg to system PATH..." -ForegroundColor Yellow
        [Environment]::SetEnvironmentVariable("PATH", "$currentPath;$ffmpegBin", "Machine")
        Write-Host "✅ FFmpeg added to PATH!" -ForegroundColor Green
        Write-Host "⚠️  Please restart your terminal/IDE for PATH changes to take effect." -ForegroundColor Red
    } else {
        Write-Host "✅ FFmpeg is already in PATH!" -ForegroundColor Green
    }
    
    # Test installation
    Write-Host "🧪 Testing FFmpeg installation..." -ForegroundColor Yellow
    & "$ffmpegBin\ffmpeg.exe" -version | Select-Object -First 3
    
    Write-Host ""
    Write-Host "🎉 Installation Complete!" -ForegroundColor Green
    Write-Host "📋 Next Steps:" -ForegroundColor Yellow
    Write-Host "   1. Restart your terminal/IDE" -ForegroundColor White
    Write-Host "   2. Restart your Python backend server" -ForegroundColor White
    Write-Host "   3. Try uploading MP3/MP4 files in your app" -ForegroundColor White
    
} catch {
    Write-Host "❌ Error installing FFmpeg: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "💡 Try manual installation from: https://ffmpeg.org/download.html" -ForegroundColor Yellow
}