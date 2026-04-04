[CmdletBinding()]
param(
    [string]$Output = "data/alfworld_trajs.jsonl",
    [int]$NumEpisodes = 100,
    [double]$SuccessRate = 0.3,
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $scriptDir "run_collect_benchmark.ps1") -EnvName alfworld -Output $Output -NumEpisodes $NumEpisodes -SuccessRate $SuccessRate -Seed $Seed
exit $LASTEXITCODE