# start_clients.ps1
# クライアントを自動で起動（別ウィンドウで表示）

for ($i = 1; $i -le 5; $i++) {
    Write-Host "Starting client$i..."
    Start-Process powershell -ArgumentList "-NoExit", "-Command python client.py client$i"
    Start-Sleep -Milliseconds 300
}
