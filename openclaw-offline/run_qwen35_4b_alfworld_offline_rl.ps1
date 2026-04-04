[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir

if (-not $env:OFFLINE_TRAJECTORY_STORE) {
    $env:OFFLINE_TRAJECTORY_STORE = [System.IO.Path]::GetFullPath((Join-Path $repoRoot "offline-rl\data\alfworld_trajs.jsonl"))
}

& (Join-Path $scriptDir "run_qwen35_4b_offline_rl.ps1") @ExtraArgs
exit $LASTEXITCODE