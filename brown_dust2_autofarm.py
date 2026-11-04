import os
import json
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass

import cv2
import numpy as np
import pyautogui
import keyboard

# When frozen by PyInstaller, keep user config/templates beside the EXE.
if getattr(__import__("sys"), "frozen", False):
    APP_DIR = os.path.dirname(os.path.abspath(__import__("sys").executable))
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(APP_DIR, "templates")
CONFIG_FILE = os.path.join(APP_DIR, "config.json")
os.makedirs(TEMPLATE_DIR, exist_ok=True)

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.15

DEFAULT_CONFIG = {
    "confidence": 0.86,
    "poll_interval": 0.6,
    "click_delay": 1.2,
    "dialog_wait": 2.0,
    "battle_timeout": 600,
    "step_timeout": 90,
    "templates": {
        "章节页自动进行按钮": "auto_explore.png",
        "探索奖励弹窗": "reward_screen.png",
        "奖励后战斗按钮": "battle_button.png",
        "跳过对话确认按钮": "skip_confirm.png",
        "战斗界面识别区域": "battle_screen.png",
        "自动战斗已开启图标": "auto_mode_on.png",
        "自动战斗未开启图标": "auto_mode_off.png",
        "胜利结算界面": "victory_screen.png",
        "胜利后离开按钮": "leave_button.png"
    }
}

@dataclass
class Hit:
    x: int
    y: int
    score: float

def load_config():
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    for k, v in DEFAULT_CONFIG.items():
        if k not in data:
            data[k] = v
    for k, v in DEFAULT_CONFIG["templates"].items():
        data["templates"].setdefault(k, v)
    return data

def save_config(cfg):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

class Vision:
    def __init__(self, confidence):
        self.confidence = confidence

    def screen(self):
        img = pyautogui.screenshot()
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def find(self, filename, threshold=None):
        path = os.path.join(TEMPLATE_DIR, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"缺少模板：{filename}")
        templ = cv2.imread(path, cv2.IMREAD_COLOR)
        if templ is None:
            raise RuntimeError(f"模板无法读取：{filename}")
        scr = self.screen()
        h, w = templ.shape[:2]
        sh, sw = scr.shape[:2]
        if h > sh or w > sw:
            return None
        res = cv2.matchTemplate(scr, templ, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        limit = self.confidence if threshold is None else threshold
        if score < limit:
            return None
        return Hit(loc[0] + w // 2, loc[1] + h // 2, float(score))

    def wait(self, filename, timeout, stop_event, log, threshold=None):
        started = time.time()
        while time.time() - started < timeout:
            if stop_event.is_set():
                return None
            hit = self.find(filename, threshold)
            if hit:
                log(f"识别到 {filename}，匹配度 {hit.score:.2f}")
                return hit
            time.sleep(0.5)
        return None

class Runner:
    def __init__(self, log, finished):
        self.log = log
        self.finished = finished
        self.stop_event = threading.Event()
        self.thread = None

    def start(self, runs, difficulty, cfg):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.run, args=(runs, difficulty, cfg), daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.log("已请求停止；脚本会在当前等待步骤结束时退出。")

    def wait_and_click(self, vision, filename, timeout, cfg, label=None):
        hit = vision.wait(filename, timeout, self.stop_event, self.log)
        if hit is None:
            if not self.stop_event.is_set():
                self.log(f"超时：没有识别到 {label or filename}")
            return False
        pyautogui.click(hit.x, hit.y)
        time.sleep(cfg["click_delay"])
        return True

    def run(self, runs, difficulty, cfg):
        vision = Vision(float(cfg["confidence"]))
        t = cfg["templates"]
        done = 0
        try:
            self.log(f"启动循环：难度={difficulty}，目标轮数={runs}")
            self.log("请先手动进入目标游戏卡的章节页，并保持游戏窗口可见。")
            self.log("注意：此版本从章节页的‘自动进行’按钮开始接管循环。")
            for n in range(1, runs + 1):
                if self.stop_event.is_set():
                    break
                self.log(f"===== 第 {n}/{runs} 轮 =====")
                if not self.wait_and_click(vision, t["章节页自动进行按钮"], cfg["step_timeout"], cfg, "章节页自动进行按钮"):
                    break
                reward = vision.wait(t["探索奖励弹窗"], cfg["step_timeout"], self.stop_event, self.log)
                if reward is None:
                    if not self.stop_event.is_set(): self.log("没有检测到探索奖励弹窗，停止以避免误点。")
                    break
                sw, sh = pyautogui.size()
                pyautogui.click(sw // 2, sh // 2)
                time.sleep(cfg["click_delay"])
                if not self.wait_and_click(vision, t["奖励后战斗按钮"], cfg["step_timeout"], cfg, "奖励后战斗按钮"):
                    break
                time.sleep(cfg["dialog_wait"])
                pyautogui.press("f2")
                time.sleep(1.0)
                confirm = vision.wait(t["跳过对话确认按钮"], 8, self.stop_event, self.log)
                if confirm:
                    pyautogui.click(confirm.x, confirm.y)
                    time.sleep(cfg["click_delay"])
                else:
                    self.log("未检测到跳过确认按钮；继续等待战斗界面（可能没有对话弹窗）。")
                battle = vision.wait(t["战斗界面识别区域"], cfg["step_timeout"], self.stop_event, self.log)
                if battle is None:
                    if not self.stop_event.is_set(): self.log("未识别到战斗界面，停止。")
                    break
                off = vision.find(t["自动战斗未开启图标"])
                on = vision.find(t["自动战斗已开启图标"])
                if off:
                    self.log("检测到自动战斗未开启，点击对应图标。")
                    pyautogui.click(off.x, off.y)
                    time.sleep(cfg["click_delay"])
                elif on:
                    self.log("自动战斗已开启，无需操作。")
                else:
                    self.log("无法判断自动战斗开关状态，为避免误操作不点击；请检查两张图标模板。")
                    break
                victory = vision.wait(t["胜利结算界面"], cfg["battle_timeout"], self.stop_event, self.log)
                if victory is None:
                    if not self.stop_event.is_set(): self.log("等待胜利结算超时，停止。")
                    break
                if not self.wait_and_click(vision, t["胜利后离开按钮"], cfg["step_timeout"], cfg, "胜利后离开按钮"):
                    break
                back = vision.wait(t["章节页自动进行按钮"], cfg["step_timeout"], self.stop_event, self.log)
                if back is None:
                    if not self.stop_event.is_set(): self.log("离开后未返回章节页，停止。")
                    break
                done += 1
                self.log(f"第 {n} 轮完成。")
                time.sleep(1.0)
            self.log(f"任务结束：完成 {done}/{runs} 轮。")
        except pyautogui.FailSafeException:
            self.log("触发紧急停止：鼠标移动到了屏幕左上角。")
        except Exception as e:
            self.log(f"错误：{type(e).__name__}: {e}")
        finally:
            self.finished()

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("棕色尘埃2 自动刷图助手 V2.0")
        self.root.geometry("790x650")
        self.root.minsize(730, 580)
        self.cfg = load_config()
        self.runner = Runner(self.log, self.done)
        self.build()
        self.refresh()

    def build(self):
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="棕色尘埃2 自动刷图助手 V2.0", font=("Microsoft YaHei UI", 16, "bold")).pack(anchor="w")
        ttk.Label(outer, text="探索奖励 → 战斗 → 胜利 → 离开 → 下一轮", foreground="#555").pack(anchor="w", pady=(2, 10))
        settings = ttk.LabelFrame(outer, text="运行设置", padding=10)
        settings.pack(fill="x")
        row = ttk.Frame(settings); row.pack(fill="x", pady=3)
        ttk.Label(row, text="游戏卡类型：").pack(side="left")
        self.category = tk.StringVar(value="角色游戏卡")
        ttk.Combobox(row, textvariable=self.category, values=["角色游戏卡", "剧情游戏卡"], state="readonly", width=14).pack(side="left", padx=(0, 16))
        ttk.Label(row, text="难度：").pack(side="left")
        self.difficulty = tk.StringVar(value="困难")
        ttk.Combobox(row, textvariable=self.difficulty, values=["困难", "极难"], state="readonly", width=10).pack(side="left", padx=(0, 16))
        ttk.Label(row, text="循环次数：").pack(side="left")
        self.runs = tk.StringVar(value="10")
        ttk.Entry(row, textvariable=self.runs, width=8).pack(side="left")
        row2 = ttk.Frame(settings); row2.pack(fill="x", pady=5)
        ttk.Label(row2, text="识别阈值：").pack(side="left")
        self.threshold = tk.DoubleVar(value=float(self.cfg["confidence"]))
        ttk.Scale(row2, from_=0.70, to=0.98, variable=self.threshold, orient="horizontal", length=200).pack(side="left", padx=6)
        self.threshold_label = ttk.Label(row2, text=f"{self.threshold.get():.2f}")
        self.threshold_label.pack(side="left")
        self.threshold.trace_add("write", lambda *_: self.threshold_label.config(text=f"{self.threshold.get():.2f}"))
        tpl_frame = ttk.LabelFrame(outer, text="识别模板录制（首次使用必须完成）", padding=8)
        tpl_frame.pack(fill="both", expand=True, pady=10)
        self.tpl_list = tk.Listbox(tpl_frame, height=11, font=("Consolas", 10))
        self.tpl_list.pack(side="left", fill="both", expand=True)
        side = ttk.Frame(tpl_frame); side.pack(side="left", fill="y", padx=10)
        ttk.Button(side, text="录制选中模板", command=self.capture_selected).pack(fill="x", pady=3)
        ttk.Button(side, text="打开模板目录", command=lambda: os.startfile(TEMPLATE_DIR)).pack(fill="x", pady=3)
        ttk.Button(side, text="打开配置文件", command=lambda: os.startfile(CONFIG_FILE)).pack(fill="x", pady=3)
        ttk.Button(side, text="刷新模板状态", command=self.refresh).pack(fill="x", pady=3)
        controls = ttk.Frame(outer); controls.pack(fill="x", pady=3)
        self.start_btn = ttk.Button(controls, text="开始自动刷图", command=self.start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(controls, text="停止", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=8)
        log_frame = ttk.LabelFrame(outer, text="运行日志", padding=8)
        log_frame.pack(fill="both", expand=True)
        self.logs = tk.Text(log_frame, height=9, state="disabled", wrap="word")
        self.logs.pack(fill="both", expand=True)
        ttk.Label(outer, text="安全停止：点击‘停止’或将鼠标移到屏幕左上角。首次请只测试 1 轮。", foreground="#8a4b08").pack(anchor="w", pady=(7, 0))

    def refresh(self):
        self.tpl_list.delete(0, "end")
        self.template_keys = list(self.cfg["templates"].keys())
        for key in self.template_keys:
            fn = self.cfg["templates"][key]
            exists = os.path.isfile(os.path.join(TEMPLATE_DIR, fn))
            self.tpl_list.insert("end", ("✓ " if exists else "缺少 ") + key + "  [" + fn + "]")

    def capture_selected(self):
        selection = self.tpl_list.curselection()
        if not selection:
            messagebox.showinfo("提示", "请先选择一项模板。")
            return
        key = self.template_keys[selection[0]]
        filename = self.cfg["templates"][key]
        messagebox.showinfo("录制模板", f"请先把游戏切换到【{key}】对应的界面。\n点击确定后，程序会截图并让你框选一个小而清晰的区域。\n按 Enter 保存，按 C 取消。")
        self.root.iconify()
        self.root.after(1200, lambda: self._capture_roi(key, filename))

    def _capture_roi(self, key, filename):
        try:
            shot = pyautogui.screenshot()
            frame = cv2.cvtColor(np.asarray(shot), cv2.COLOR_RGB2BGR)
            roi = cv2.selectROI(f"框选模板：{key}（Enter保存 / C取消）", frame, showCrosshair=True, fromCenter=False)
            cv2.destroyAllWindows()
            x, y, w, h = [int(v) for v in roi]
            self.root.deiconify()
            if w < 8 or h < 8:
                self.log(f"取消录制：{key}")
                return
            out = os.path.join(TEMPLATE_DIR, filename)
            # Use imencode + tofile for reliable Unicode paths on Windows.
            ok, encoded = cv2.imencode(".png", frame[y:y+h, x:x+w])
            if not ok:
                raise RuntimeError("OpenCV 无法编码模板图片。")
            encoded.tofile(out)
            if not os.path.isfile(out) or os.path.getsize(out) == 0:
                raise RuntimeError("模板文件没有成功写入，请检查 templates 文件夹权限和磁盘空间。")
            self.log(f"已保存模板：{filename}")
            self.refresh()
        except Exception as e:
            self.root.deiconify()
            messagebox.showerror("录制失败", str(e))

    def log(self, msg):
        self.root.after(0, self._log_ui, msg)

    def _log_ui(self, msg):
        self.logs.configure(state="normal")
        self.logs.insert("end", time.strftime("[%H:%M:%S] ") + msg + "\n")
        self.logs.see("end")
        self.logs.configure(state="disabled")

    def start(self):
        try:
            n = int(self.runs.get())
            if n < 1 or n > 5000:
                raise ValueError
        except ValueError:
            messagebox.showerror("输入错误", "循环次数请输入 1 到 5000 的整数。")
            return
        missing = [fn for fn in self.cfg["templates"].values() if not os.path.isfile(os.path.join(TEMPLATE_DIR, fn))]
        if missing:
            messagebox.showwarning("模板未完成", "请先录制所有模板：\n" + "\n".join(missing))
            return
        self.cfg["confidence"] = float(self.threshold.get())
        save_config(self.cfg)
        messagebox.showwarning("开始前确认", "请确认：游戏已手动进入目标游戏卡的章节页面；所选难度与游戏内当前难度一致；游戏窗口可见且未被遮挡。\n\n首次务必设置 1 轮，观察每一步是否正确。")
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.runner.start(n, self.difficulty.get(), self.cfg)

    def stop(self):
        self.runner.stop()

    def done(self):
        self.root.after(0, lambda: (self.start_btn.configure(state="normal"), self.stop_btn.configure(state="disabled")))

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
