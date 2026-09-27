# JARVIS Database Backup & Disaster Recovery Strategy

This document describes backup, restore, automated scheduling, and disaster recovery procedures for the JARVIS PostgreSQL database.

---

## 1. Quick Manual Backup

### Dockerized Environment
Run from the project root:
```bash
# Create timestamped backup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
docker exec -t jarvis_postgres pg_dump -U jarvis_user -F c -b -v -f /tmp/jarvis_backup_${TIMESTAMP}.dump jarvis_db
docker cp jarvis_postgres:/tmp/jarvis_backup_${TIMESTAMP}.dump ./backups/
docker exec jarvis_postgres rm /tmp/jarvis_backup_${TIMESTAMP}.dump
```

### Local / Native PostgreSQL
```bash
# Custom-format compressed backup (recommended)
pg_dump -U jarvis_user -h localhost -p 5432 -F c -b -v -f "jarvis_backup_$(date +%Y%m%d_%H%M%S).dump" jarvis_db

# Plain SQL format
pg_dump -U jarvis_user -h localhost -p 5432 -F p -v -f "jarvis_backup_$(date +%Y%m%d_%H%M%S).sql" jarvis_db
```

---

## 2. Quick Restore Procedure

### Restoring from Custom Format (.dump)
```bash
# 1. Terminate active connections
psql -U jarvis_user -d postgres -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = 'jarvis_db' AND pid <> pg_backend_pid();"

# 2. Drop and recreate database
psql -U jarvis_user -d postgres -c "DROP DATABASE IF EXISTS jarvis_db;"
psql -U jarvis_user -d postgres -c "CREATE DATABASE jarvis_db OWNER jarvis_user;"

# 3. Restore using pg_restore
pg_restore -U jarvis_user -h localhost -p 5432 -d jarvis_db -v "jarvis_backup_YYYYMMDD_HHMMSS.dump"

# 4. Verify Alembic migration version
cd backend
alembic current
```

---

## 3. Automated Daily Backup (Cron / Windows Task Scheduler)

### Linux Crontab (Daily at 02:00 UTC)
Create a script `/opt/jarvis/scripts/backup.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail

BACKUP_DIR="/var/backups/jarvis"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=14

mkdir -p "${BACKUP_DIR}"

# Dump database
pg_dump -U jarvis_user -h localhost -F c -b -f "${BACKUP_DIR}/jarvis_${TIMESTAMP}.dump" jarvis_db

# Delete backups older than retention window
find "${BACKUP_DIR}" -type f -name "jarvis_*.dump" -mtime +${RETENTION_DAYS} -delete

echo "[$(date)] Backup completed: jarvis_${TIMESTAMP}.dump"
```

Add to cron:
```bash
0 2 * * * /opt/jarvis/scripts/backup.sh >> /var/log/jarvis_backup.log 2>&1
```

### Windows PowerShell Scheduled Task
```powershell
$Date = Get-Date -Format "yyyyMMdd_HHmmss"
$BackupFile = "C:\jarvis_backups\jarvis_$Date.dump"
& "C:\Program Files\PostgreSQL\16\bin\pg_dump.exe" -U jarvis_user -h localhost -F c -b -f $BackupFile jarvis_db

# Purge backups older than 14 days
Get-ChildItem -Path "C:\jarvis_backups" -Filter "jarvis_*.dump" | Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-14) } | Remove-Item
```

---

## 4. Retention & Storage Policy

- **Daily Backups:** Retained for 14 days on local volume.
- **Weekly Backups:** Transferred to secondary storage / S3 / cloud bucket and retained for 8 weeks.
- **Monthly Backups:** Retained for 12 months.
- **Encryption:** Backup files contain sensitive user context and memories. When storing offsite, encrypt backups using GPG or age:
  ```bash
  gpg --symmetric --cipher-algo AES256 jarvis_backup_YYYYMMDD.dump
  ```
