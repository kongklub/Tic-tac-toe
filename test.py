import pyautogui
import time

# รอ 3 วินาทีเพื่อให้ผู้ใช้มีเวลาไปยังตำแหน่งที่ต้องการ
time.sleep(3)

# เช็คพิกัดปัจจุบันของเมาส์
current_x, current_y = pyautogui.position()

# แสดงพิกัดของเมาส์
print(f"ตำแหน่งปัจจุบันของเมาส์: (x: {current_x}, y: {current_y})")

# ตัวอย่างการใช้งานเพื่อย้ายและคลิก
# กำหนดตำแหน่งใหม่ที่ต้องการ
new_x, new_y = 500, 400

# ย้ายเมาส์ไปยังตำแหน่งใหม่
pyautogui.moveTo(new_x, new_y, duration=1)

# คลิกเมาส์ซ้ายที่ตำแหน่งนั้น
pyautogui.click()
