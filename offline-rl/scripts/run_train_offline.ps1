[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONFAULTHANDLER = "1"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$trainScript = Join-Path $scriptDir "train_offline.py"

$dataPath = if ($env:OFFLINE_TRAIN_DATA) { $env:OFFLINE_TRAIN_DATA } else { "data/osworld_trajs.jsonl" }
$algoName = if ($env:OFFLINE_TRAIN_ALGO) { $env:OFFLINE_TRAIN_ALGO } else { "iql" }
$steps = if ($env:OFFLINE_TRAIN_STEPS) { $env:OFFLINE_TRAIN_STEPS } else { "500" }
$batchSize = if ($env:OFFLINE_TRAIN_BATCH_SIZE) { $env:OFFLINE_TRAIN_BATCH_SIZE } else { "32" }
$learningRate = if ($env:OFFLINE_TRAIN_LR) { $env:OFFLINE_TRAIN_LR } else { "3e-4" }
$stateDim = if ($env:OFFLINE_TRAIN_STATE_DIM) { $env:OFFLINE_TRAIN_STATE_DIM } else { "128" }
$actionDim = if ($env:OFFLINE_TRAIN_ACTION_DIM) { $env:OFFLINE_TRAIN_ACTION_DIM } else { "128" }
$hiddenDim = if ($env:OFFLINE_TRAIN_HIDDEN_DIM) { $env:OFFLINE_TRAIN_HIDDEN_DIM } else { "128" }
$deviceName = if ($env:OFFLINE_TRAIN_DEVICE) { $env:OFFLINE_TRAIN_DEVICE } else { "cuda" }

function Get-PythonCommand {
    if ($env:PYTHON_BIN) {
        return @($env:PYTHON_BIN)
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return @($python.Source)
    }

    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        return @($py.Source, "-3")
    }

    throw "No Python launcher found. Set PYTHON_BIN, or install python/py on PATH."
}

$pythonCommand = @(Get-PythonCommand)
$pythonExe = $pythonCommand[0]
$pythonArgs = @()
if ($pythonCommand.Count -gt 1) {
    $pythonArgs += $pythonCommand[1..($pythonCommand.Count - 1)]
}

$pythonArgs += @(
    $trainScript,
    "--data", $dataPath,
    "--algo", $algoName,
    "--steps", $steps,
    "--batch-size", $batchSize,
    "--lr", $learningRate,
    "--state-dim", $stateDim,
    "--action-dim", $actionDim,
    "--hidden-dim", $hiddenDim,
    "--device", $deviceName
)

if ($env:OFFLINE_TRAIN_OUTPUT) {
    $pythonArgs += @("--output", $env:OFFLINE_TRAIN_OUTPUT)
}

if ($ExtraArgs) {
    $pythonArgs += $ExtraArgs
}

& $pythonExe @pythonArgs
exit $LASTEXITCODE