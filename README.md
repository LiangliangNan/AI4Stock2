需要安装的packages:
```bash
pip -m install matplotlib pandas akshare pyarrow seaborn fire pyqlib yaml tables torch
```

Qlib 数据缓存依赖 Redis
当你使用 build_alpha158_handler 或其他 DatasetH / DataHandler 时，Qlib 默认会启用缓存机制，尤其是 lazy_load + cache 模式。
Qlib 会尝试连接本地 Redis（默认端口 6379）来锁定数据、缓存中间结果、防止多进程冲突。

## 安装并启动本地 Redis：

### 1. Ubuntu/Debian
```bash
sudo apt update
sudo apt install redis-server
sudo systemctl enable redis-server
sudo systemctl start redis-server
```

### 2. macOS (brew)
```bash
brew install redis
brew services start redis
```