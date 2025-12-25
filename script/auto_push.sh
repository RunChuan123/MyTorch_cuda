#!/bin/bash
# auto_git_push.sh - 自动提交并推送 Git 更改

# 配置变量
COMMIT_MESSAGE="Auto commit: $(date '+%Y-%m-%d %H:%M:%S')"
BRANCH_NAME=$(git branch --show-current)
REMOTE_NAME="origin"  # 默认远程仓库名

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_info() { echo -e "${YELLOW}[INFO]${NC} $1"; }

# 检查是否在 Git 仓库中
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    print_error "当前目录不是 Git 仓库！"
    exit 1
fi

# 检查是否有未提交的更改
if git diff-index --quiet HEAD --; then
    print_info "没有需要提交的更改。"
    exit 0
fi

print_info "当前分支: $BRANCH_NAME"
print_info "提交信息: $COMMIT_MESSAGE"
echo ""

# 1. 添加所有更改
print_info "添加所有更改..."
git add .
if [ $? -ne 0 ]; then
    print_error "添加文件失败！"
    exit 1
fi
print_success "文件添加完成"

# 2. 提交更改
print_info "提交更改..."
git commit -m "$COMMIT_MESSAGE"
if [ $? -ne 0 ]; then
    print_error "提交失败！"
    exit 1
fi
print_success "提交完成"

# 3. 拉取远程最新更改（避免冲突）
# print_info "拉取远程最新更改..."
# git pull $REMOTE_NAME $BRANCH_NAME
# if [ $? -ne 0 ]; then
#     print_error "拉取失败，可能存在冲突！"
#     exit 1
# fi
# print_success "拉取完成"



# 4. 推送到远程仓库
print_info "推送到远程仓库..."

MAX_RETRY=3
retry_count=0
push_success=false

while [ $retry_count -lt $MAX_RETRY ] && [ "$push_success" = false ];do
    ((retry_count++))
    if [ $retry_count -gt 1 ]; then 
        print_info "第 $retry_count 次尝试推送"
    fi
    git push $REMOTE_NAME $BRANCH_NAME
    push_exit_code=$?
    if [ $push_exit_code -eq 0 ]; then
        print_success "✅ 推送成功！"
        push_success=true
    else
        print_info "第 $retry_count 次推送失败 (退出码: $push_exit_code)"
        if [ $retry_count -lt $MAX_RETRY ] ; then
            print_info "等待三秒再试"
            sleep 3
        else
            print_error "🚨 推送失败！已尝试 $MAX_RETRY 次"
            
            # 显示更详细的错误信息
            print_info "调试信息:"
            echo "远程仓库: $REMOTE_NAME"
            echo "分支: $BRANCH_NAME"
            echo "当前分支: $(git branch --show-current)"
            echo "远程状态:"
            git remote -v
            exit 1
        fi
    fi
done

# git push $REMOTE_NAME $BRANCH_NAME
# if [ $? -eq 0 ]; then
#     print_success "✅ 推送成功！"
# else
#     print_error "推送失败！"
#     exit 1
# fi

# 5. 显示推送后的状态
echo ""
print_info "最近一次提交:"
git log --oneline -1