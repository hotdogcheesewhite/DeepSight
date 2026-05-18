#!/bin/bash

DIR="/mnt/nas-data-1/zhanglingjun.zlj1/data/bench2drive-full"
cd "$DIR" || exit 1

# 获取CPU核心数作为默认并发数
CONCURRENT=128
# 获取文件列表
mapfile -t files < <(ls -1 *.tar.gz 2>/dev/null)
total=${#files[@]}

echo "找到 $total 个tar.gz文件，使用 $CONCURRENT 个并发进程"

# 定义并发处理函数
process_file() {
    local file="$1"
    local index="$2"
    local total="$3"
    
    echo "[$index/$total] 正在解压: $file"
    if tar -xzf "$file"; then
        rm "$file"
        echo "[$index/$total] 成功解压并删除: $file"
    else
        echo "[$index/$total] 解压失败，保留文件: $file"
    fi
}

# 并发处理文件
for i in "${!files[@]}"; do
    # 在后台启动子进程处理文件
    process_file "${files[$i]}" "$((i+1))" "$total" &
    
    # 控制并发数量
    if (( (i % CONCURRENT) == (CONCURRENT - 1) )); then
        wait
    fi
done

# 等待所有后台进程完成
wait

echo "处理完成！"