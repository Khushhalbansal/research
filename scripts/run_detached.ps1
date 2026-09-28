# Launches `lrmc run-queue` detached from the current PowerShell/AnyDesk
# session, so it survives a disconnect: Start-Process with a hidden window
# and output redirected to a log file (detached processes started this way
# keep running after an AnyDesk disconnect as long as the Windows session
# itself isn't logged off -- the usual case for a remote-desktop-style
# disconnect, as opposed to a full sign-out).
#
# Usage: powershell -File scripts\run_detached.ps1 [-Queue path\to\queue.yaml]
param(
    [string]$Queue = "experiments/queue.yaml",
    [string]$LogDir = "logs"
)

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "run_queue_$Timestamp.log"
$ErrFile = "$LogFile.err"

$PythonExe = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }

Write-Host "Starting detached queue run. Log: $LogFile"
$Process = Start-Process -FilePath $PythonExe `
    -ArgumentList @("-m", "lrmc.cli.main", "run-queue", $Queue) `
    -RedirectStandardOutput $LogFile `
    -RedirectStandardError $ErrFile `
    -WindowStyle Hidden `
    -PassThru

Write-Host "Started with PID $($Process.Id)."
Write-Host "Check progress any time with: $PythonExe -m lrmc.cli.main status"
Write-Host "Or tail the log: Get-Content -Wait -Tail 30 $LogFile"

# Optional, stronger guarantee: register as a Scheduled Task so the run
# survives a full logoff/reboot, not just a disconnected AnyDesk session
# (Start-Process above is enough for a plain disconnect; a Scheduled Task
# with "Run whether user is logged on or not" is the belt-and-suspenders
# option if IT policy logs the session off after a period of inactivity).
# Uncomment to use instead of the Start-Process call above:
#
# $Action  = New-ScheduledTaskAction -Execute $PythonExe -Argument "-m lrmc.cli.main run-queue $Queue" -WorkingDirectory $RepoRoot
# $Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date)
# Register-ScheduledTask -TaskName "LRMC-RunQueue" -Action $Action -Trigger $Trigger -RunLevel Highest -Force
# Start-ScheduledTask -TaskName "LRMC-RunQueue"
# Write-Host "Registered and started Scheduled Task 'LRMC-RunQueue' (survives logoff/reboot)."
