[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$bashScript = [System.IO.Path]::GetFullPath((Join-Path $scriptDir "run_qwen35_4b_offline_rl.sh"))

if (-not $env:OFFLINE_TRAJECTORY_STORE) {
    $env:OFFLINE_TRAJECTORY_STORE = [System.IO.Path]::GetFullPath((Join-Path $repoRoot "offline-rl\data\osworld_trajs.jsonl"))
}
if (-not $env:OFFLINE_MODE) {
    $env:OFFLINE_MODE = "step"
}
if (-not $env:OFFLINE_N_SAMPLES_PER_PROMPT) {
    $env:OFFLINE_N_SAMPLES_PER_PROMPT = "1"
}

function Quote-Bash([string]$Value) {
    if ($null -eq $Value) {
        return "''"
    }

    $escaped = $Value -replace "'", "'`"'`"'"
    return "'{0}'" -f $escaped
}

function Convert-ToWslPath([string]$PathValue) {
    $resolved = [System.IO.Path]::GetFullPath($PathValue)
    $drive = $resolved.Substring(0, 1).ToLowerInvariant()
    $suffix = $resolved.Substring(2) -replace '\\', '/'
    return "/mnt/{0}{1}" -f $drive, $suffix
}

$quotedArgs = @()
foreach ($arg in $ExtraArgs) {
    $quotedArgs += Quote-Bash $arg
}

$wsl = Get-Command wsl -ErrorAction SilentlyContinue
if ($wsl) {
    $bashScriptWsl = Convert-ToWslPath $bashScript
    $commandParts = @(
        "export OFFLINE_TRAJECTORY_STORE=$(Quote-Bash (Convert-ToWslPath $env:OFFLINE_TRAJECTORY_STORE))",
        "export OFFLINE_MODE=$(Quote-Bash $env:OFFLINE_MODE)",
        "export OFFLINE_N_SAMPLES_PER_PROMPT=$(Quote-Bash $env:OFFLINE_N_SAMPLES_PER_PROMPT)"
    )

    if ($env:OFFLINE_WEIGHT_PATH) {
        $commandParts += "export OFFLINE_WEIGHT_PATH=$(Quote-Bash (Convert-ToWslPath $env:OFFLINE_WEIGHT_PATH))"
    }
    if ($env:OFFLINE_WEIGHT_TEMPERATURE) {
        $commandParts += "export OFFLINE_WEIGHT_TEMPERATURE=$(Quote-Bash $env:OFFLINE_WEIGHT_TEMPERATURE)"
    }

    $launchCommand = "bash {0}" -f (Quote-Bash $bashScriptWsl)
    if ($quotedArgs.Count -gt 0) {
        $launchCommand = "{0} {1}" -f $launchCommand, ($quotedArgs -join " ")
    }
    $commandParts += $launchCommand

    & $wsl.Source bash -lc ($commandParts -join "; ")
    exit $LASTEXITCODE
}

$bash = Get-Command bash -ErrorAction SilentlyContinue
if ($bash) {
    Write-Warning "WSL was not found. Falling back to bash on PATH. A Linux-like runtime is still required for full slime training."
    $bashScriptPosix = $bashScript -replace '\\', '/'
    & $bash.Source $bashScriptPosix @ExtraArgs
    exit $LASTEXITCODE
}

throw "No WSL or bash runtime found. Install WSL or Git Bash, or run the .sh launcher from a Linux-like environment."