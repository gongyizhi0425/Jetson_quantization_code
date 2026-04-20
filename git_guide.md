# Git 常用操作指南

## 1. 首次上传本地项目到 GitHub

**前提**：先在 GitHub 网页上创建一个**空仓库**（不勾选 README、.gitignore），拿到仓库 URL。

```bash
git init                    # 初始化本地仓库
git branch -M main          # 确保默认分支名为 main（而非 master）
git add .                   # 将所有文件添加到暂存区
git commit -m "first commit"  # 提交到本地仓库
git remote add origin https://github.com/yourname/yourrepo.git  # 关联远程仓库
git push -u origin main     # 推送到远程 main 分支（-u 设置上游跟踪）
```

**说明**：
- `git add .` 将工作区所有文件加入暂存区，`git commit` 将暂存区内容提交到本地仓库。
- `git push -u origin main` 将本地 main 分支推送到远程。如果远程还没有 main 分支，会自动创建。
- `-u` 参数设置上游跟踪后，以后只需 `git push` 即可，不用再写 `origin main`。
- **origin main 是远程仓库的默认分支，也是最重要的。初次提交之后不要直接改动它，应该通过分支开发再合并。**

---

## 2. 本地已有仓库，推送到新建的远程仓库

适用场景：本地已经有 git 仓库和提交记录，只需关联远程。
注意，这个是直接开一个新仓库，而不是在已有仓库的基础上添加远程仓库。
```bash
git remote add origin https://github.com/yourname/yourrepo.git
git push -u origin main
```

---

## 3. 创建分支开发，再合并到 main

```bash
# 创建并切换到新分支
git checkout -b yourbranch
！目前本地的工作分支是working
# 查看当前分支
git branch --show-current
注意，永远不要在 main 分支上直接开发，而是创建新分支！防止丢失重要代码。

# 在新分支上正常开发、提交
git add .
git commit -m "feat: 新功能描述"


# 推送新分支到远程
git push -u origin yourbranch

！注意此时相当于在远程自动创建了origin/yourbranch分支，并且与本地yourbranch分支关联。
！目前这个远程的个人分支叫Eazon

git push -u origin working:Eazon

！这个命令的意思是，将本地的working分支推送到远程的Eazon分支，并且建立关联。 必须先有本地分支：working:Eazon 中冒号左边是本地分支名，右边是远程分支名。本地分支必须已经存在，远程分支可以不存在（会自动创建）。
-u 的作用是设置上游跟踪：设置后，以后在 working 分支上直接 git push 就会自动推送到 origin/Eazon，不用每次都写完整命令。


# 开发完成后，切回 main 并合并
git checkout main
git merge yourbranch

# 合并后推送 main
git push origin main

git branch -vv # 查看本地分支与远程分支的关联关系
```

---
## 4. 日常提交流程 最常用！！   1-3是首次使用时的初始化

```bash
git checkout working # 每天第一步，先到本地的工作分支
写代码、改文件
（可选）git pull origin main        # 把最新 main 拉到 working，保持同步
git add -A                  # 暂存所有更改（新增、修改、删除）
git commit -m "描述你的更改"

第一次：git config push.default upstream  # 设置默认推送分支为上游分支，以后 git push 就不用写 origin/Eazon 了

git push                    ## 已设置 -u，自动推到 origin/Eazon
这是主要的日常工作流。我们上传到远端的代码全是在Eazon分支上，main分支是稳定的，不直接修改。
检查整个仓库的状况：git status
git branch -vv
git branch -a


！而Eazon要合并到 main 时 → 在 GitHub 网页上发 Pull Request
打开 GitHub 仓库页面
点 "Compare & pull request"（或 Pull requests → New）  这里所有东西都是远程的，只关心远程个人和远程main之间的比较，线上判断是否可以合并
选择 Eazon → main
写标题和描述，点 Create pull request
队友 review 后点 Merge pull request
这样远程 main 由 GitHub 网页操作合并，你本地和命令行完全不碰 main。


！另一种方法，当我们是仓库主人时：此时如果所有功能都稳定了，可以合并到 本地和远程的main：

git checkout main  #切换到本地的main
git pull origin main  #拉取远程main分支(这是多人协作时必须的，防止冲突)，这会直接更新本地mian
git merge working  #把working的改动合并进入本地main，本地main变了
git push origin main  #推送本地main到远程main
！这是最后的最后要做的!

```
---

## 5. 常用查看命令

```bash
git remote -v               # 查看远程仓库地址（确认你在哪个仓库）
git branch --show-current   # 查看当前分支
git branch -a               # 查看所有分支（含远程）
git log --oneline -10       # 查看最近 10 条提交历史
git status                  # 查看工作区状态（哪些文件改了、哪些已暂存）
git diff                    # 查看未暂存的更改内容
```


## 6. 常见问题
转私有后怎么邀请别人
打开仓库页面 → Settings → 左侧 Collaborators（或 Collaborators and teams）
点 Add people
输入对方的 GitHub 用户名或邮箱
选择权限：
Read — 只能看代码
Write — 可以推送代码
Admin — 完全管理权限
对方会收到邮件邀请，接受后就能访问了。
