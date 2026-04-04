[CmdletBinding()]
param(
    [string]$Output = "data/androidworld_trajs.jsonl",
    [int]$NumEpisodes = 100,
    [double]$SuccessRate = 0.3,
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $scriptDir "run_collect_benchmark.ps1") -EnvName androidworld -Output $Output -NumEpisodes $NumEpisodes -SuccessRate $SuccessRate -Seed $Seed
exit $LASTEXITCODE