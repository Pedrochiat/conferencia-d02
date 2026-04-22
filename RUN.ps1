$ErrorActionPreference = "Stop"

$BaseDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $BaseDir

$VenvDir = Join-Path $BaseDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

if (!(Test-Path $VenvPython)) {
  if (Get-Command py -ErrorAction SilentlyContinue) {
    py -3 -m venv $VenvDir
  }
  elseif (Get-Command python -ErrorAction SilentlyContinue) {
    python -m venv $VenvDir
  }
  else {
    throw "Python nao encontrado no sistema."
  }
}

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r (Join-Path $BaseDir "requirements.txt")

$Script = Join-Path $BaseDir "scripts\python\21_run_modelo_d02.py"
& $VenvPython $Script
if ($LASTEXITCODE -ne 0) {
  throw "Falha ao executar o processamento do caso D02."
}

Write-Host "Processamento concluido com sucesso."
