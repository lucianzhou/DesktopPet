# DesktopPet — 爆米花

一个原生 macOS 桌面悬浮宠物。爆米花会安静地趴在桌面上睡觉，被点击后醒来伸懒腰、蹲坐，并用头和眼神追随鼠标。

## 互动方式

- 默认：趴着睡觉，播放轻微呼吸动画。
- 单击：醒来并完成一次猫式前伸，随后蹲坐。
- 蹲坐时移动鼠标：头部与视线按 16 个方向跟随鼠标。
- 蹲坐时双击：站立 5 秒，然后恢复蹲坐。
- 5 分钟没有直接点击互动：重新睡觉。鼠标移动不会重置计时。
- 拖动爆米花：移动它在桌面上的位置，位置会自动保存。
- 右键：立即睡觉或退出应用。

## 系统要求

- macOS 13 或更新版本
- Apple Silicon 或 Intel Mac（从源码构建时使用当前机器架构）
- Swift 6.2 或更新版本

不要求完整 Xcode；安装 Apple Command Line Tools 即可构建。

## 构建与运行

```bash
git clone https://github.com/lucianzhou/DesktopPet.git
cd DesktopPet
./Scripts/run.sh
```

构建后的应用位于：

```text
dist/DesktopPet.app
```

也可以分别执行：

```bash
./Scripts/test.sh
./Scripts/build-app.sh
open dist/DesktopPet.app
```

首次打开本地构建的应用时，如果 macOS 显示安全提醒，可在“系统设置 → 隐私与安全性”中允许打开。

## 工程结构

```text
Sources/DesktopPetCore/   可测试的宠物状态与视线模型
Sources/DesktopPet/       AppKit 悬浮窗口、事件处理和精灵动画
Assets/                   爆米花标准及互动精灵图集
Sources/DesktopPetCoreChecks/  无界面的核心状态自动检查
Scripts/                  .app 构建与运行脚本
Support/Info.plist        macOS 应用元数据
```

## 技术说明

- 原生 Swift + AppKit，无第三方运行时依赖。
- 透明、无边框、置顶的 `NSPanel`，可跨桌面空间显示。
- 使用一张标准 8×11 精灵图集和一张互动 8×5 精灵图集。
- 应用以 accessory 模式运行，不占用 Dock 图标；通过右键菜单退出。
- 本地构建使用 ad-hoc code signing，便于直接运行和测试。

## 资产说明

宠物形象与动画素材基于爆米花的照片定制。请勿在未经主人许可的情况下将素材用于其他商业角色或训练数据。
