[CmdletBinding()]
param(
    [string]$Output = "data/webarena_trajs.jsonl",
    [int]$NumEpisodes = 100,
    [double]$SuccessRate = 0.3,
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $scriptDir "run_collect_benchmark.ps1") -EnvName webarena -Output $Output -NumEpisodes $NumEpisodes -SuccessRate $SuccessRate -Seed $Seed
exit $LASTEXITCODE