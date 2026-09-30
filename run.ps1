<#!
.SYNOPSIS
    Run the Denoise & Separate CLI with the repository's virtual environment.

.EXAMPLE
    .\run.ps1 doctor
    .\run.ps1
#>
[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Arguments
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "The project's Python environment was not found at '$Python'. Create it first, then install requirements.txt."
}

& $Python (Join-Path $Root "run.py") @Arguments
exit $LASTEXITCODE
