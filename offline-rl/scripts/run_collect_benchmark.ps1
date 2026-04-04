[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("osworld", "androidworld", "webarena", "alfworld")]
    [string]$EnvName,

    [string]$Output,

    [int]$NumEpisodes = 100,

    [double]$SuccessRate = 0.3,

    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$collectorScript = Join-Path $scriptDir "collect_from_benchmark.py"

if (-not $Output) {
    $Output = "data/{0}_trajs.jsonl" -f $EnvName
}

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
    $collectorScript,
    "--env", $EnvName,
    "--output", $Output,
    "--num-episodes", $NumEpisodes.ToString(),
    "--success-rate", $SuccessRate.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "--seed", $Seed.ToString()
)

& $pythonExe @pythonArgs
exit $LASTEXITCODE