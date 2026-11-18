param([Parameter(Position = 0)][string]$Task = "help")

switch ($Task) {
    "test"   { python -m pytest }
    "lint"   { python -m ruff check . }
    "format" { python -m ruff format .; python -m ruff check --fix . }
    "check"  { python -m ruff check .; python -m ruff format --check .; python -m pytest }
    default  { Write-Host "Usage: .\tasks.ps1 [test | lint | format | check]" }
}