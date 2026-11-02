from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import telebot
import datetime
import subprocess
import re

import markdown
from bs4 import BeautifulSoup
import PIL
from pynvml import *
import os, psutil, numpy as np # psutil may need to be installed
def usage():
     process = psutil.Process(os.getpid())
     return process.memory_info()[0] / float(2 ** 20)

def format_memory_info(info):
    total_mb = info.total / (1024 ** 2)
    free_mb = info.free / (1024 ** 2)
    used_mb = info.used / (1024 ** 2)

    return f"total: {total_mb:.2f} MB\nfree: {free_mb:.2f} MB\nused: {used_mb:.2f} MB"
def get_gpu_info():
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        utilization = nvmlDeviceGetUtilizationRates(handle)
        temperature = nvmlDeviceGetTemperature(handle, 0)
        info = nvmlDeviceGetMemoryInfo(handle)
        ans = f'GPU Utilization: {utilization.gpu}%\nGPU Temperature: {temperature}°C\n{format_memory_info(info)}'
        return ans
    except Exception as e:
        print(f"Error: {e}")
        return f"Error: {e}"

if __name__ == '__main__':
    bot = telebot.TeleBot(os.environ["LEGACY_OLEG_TELEGRAM_BOT_TOKEN"])
    WORK = True
    model_name = "Qwen/Qwen2.5-Coder-14B-Instruct-GPTQ-Int4"
    admins = [304622290]
    SAVE_STORY = False

    torch.cuda.empty_cache()
    varname = ["Qwen/Qwen2.5-Coder-7B-Instruct","Qwen/Qwen2.5-Coder-14B-Instruct-GPTQ-Int4","Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4"]

    model = AutoModelForCausalLM.from_pretrained(
         model_name,
         torch_dtype="auto",
         device_map = {"":"cuda"}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    story = {}
    def clear_story(userid):
        global story
        story[userid] = [
                        {"role": "system", "content": "Ты помощник, тебя зовут AI Олег.Оформляй ответ с использованием корректной разметки Markdown"}
                    ]

    def get_story(userid,text = False,roles = "user"):
        global story
        if story.get(userid) == None:
            story[userid] = [
                        {"role": "system", "content": "Ты помощник, тебя зовут AI Олег.Оформляй ответ с использованием корректной разметки Markdown"}
                    ]
        if text!=False:
          story[userid].append({"role": roles, "content": text})
        return story.get(userid)
    def gtime(text):
        print(datetime.datetime.now().strftime("%H:%M:%S"),': ', text)

    def getans(text,userid):
        tstart = datetime.datetime.now()
        if SAVE_STORY:
            messages = [
                {"role": "system", "content": "Ты помощник, тебя зовут AI Олег.Оформляй ответ с использованием корректной разметки Markdown"},
                {"role": "user", "content": text}
            ]
        else:
            messages = get_story(userid,text)

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        gtime(text)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        #bot.send_message(304622290, f'Токенов запроса: {len(model_inputs)}')
        gtime(2)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512*4
        )
        gtime(3)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        gtime(4)
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        get_story(userid,response,"assistant")
        #get_story(userid,text,roles = "user")
        return [response,f"Генерация ответа: {str(datetime.datetime.now() - tstart).split('.')[0]}\nСимволов:{len(response)}"]

    def sendpart(user,msgtext):
        for i in range(0, len(msgtext), 2048):
            part = msgtext[i:i+2048]
            bot.send_message(user, part)

    @bot.message_handler(commands=['login','info','kill','model','help','par','clear','save'])
    def handle_par_command(message):
        global model, tokenizer,SAVE_STORY,story
        is_admin = message.from_user.id in admins
        response = f"Такой команды нет!\nДопустимые команды:\n/info - Информацию с видеокарты\n/kill - Убить бота\n/model - Сменить модель\n/help - Допустимые команды"
        cmd = message.text.split()[0]
        args = message.text.split()[1:]
        print(cmd)
        if cmd == '/login':
            response = f"Вы отправили команду /par с аргументами: {', '.join(args)}"
            if args[0] == "oleg":
                admins.append(message.from_user.id)
                response = "Вы авторизованы"
            else:
                response = f"Вы отправили команду /par с аргументами: {', '.join(args)}"
        elif (cmd == '/help') or (cmd == '/par'):
            response = f"Допустимые команды:\n/save - Режим диалога\n/clear - Отчистить историю\n/info - Информацию с видеокарты\n/kill - Убить бота\n/model - Сменить модель\n/help - Допустимые команды"
        elif cmd == '/clear':
            clear_story(message.from_user.id)
            response = f"История отчищена"
        ##ТОЛЬКО АВТОРИЗОВАННЫМ
        if is_admin:
            if cmd == '/info':
                cuda = torch.cuda.memory_summary(device="cuda")
                print(cuda)
                response = f"Модель: {model_name}\n"
                response += f"История: {len(get_story(message.from_user.id))}\n"
                #response += f"ВСЕГО: {cuda}\n"
                response += f"Система: {usage()} MB\n"
                response += get_gpu_info()
            elif cmd == '/kill':
                response = f"Выключение..."
                print(response)
                bot.reply_to(message, response)
                bot.stop_polling()
                torch.cuda.empty_cache()
                sys.exit()
                quit()
                exit()
            elif cmd == '/model':
                response = f"Выбери модель:\n1: Qwen/Qwen2.5-Coder-7B-Instruct\n2: Qwen/Qwen2.5-Coder-14B-Instruct-GPTQ-Int4\n3: Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4"
                bot.register_next_step_handler(message,set_model)
            elif cmd == '/save':
                if SAVE_STORY:
                    SAVE_STORY = False
                else:
                    SAVE_STORY = True
                response = f"Диалог: {SAVE_STORY}"

        bot.reply_to(message, response)
        print(response)

    @bot.message_handler(content_types=['text'])
    def start(message):
        try:
            bot.send_message(304622290, f'Думаю....\nДлина сообщения: {len(message.text)}')
            who = message.from_user.first_name

            print(f"{who}\n{message.text}")

            if (message.reply_to_message!=None):
                message.text = f"{message.reply_to_message.text}\n{message.text}"

            ans = getans(message.text,message.from_user.id)
            print(ans)

            # Проверка на правильность форматирования Markdown
            bot.send_message(message.from_user.id, ans[0], parse_mode="Markdown")
            bot.send_message(message.from_user.id, ans[1])
            if message.from_user.id != int(os.environ["LEGACY_OLEG_ADMIN_ID"]):
                bot.send_message(304622290, f"{who}\n{message.text}")
                bot.send_message(304622290, f"{who}\n{ans}")

        except telebot.apihelper.ApiTelegramException as e:
            print(f"API error: {e}")
            # Отправляем сообщение пользователю о проблеме
            bot.send_message(message.from_user.id, "Произошла ошибка при отправке сообщения. Попробуйте еще раз.")
            sendpart(message.from_user.id,ans)

    def set_model(message):
        global model_name,model,tokenizer
        if isinstance(int(message.text), int):
            if int(message.text)>=1 and int(message.text)<=3:
                model_name = varname[int(message.text)-1]
                bot.send_message(message.from_user.id,f'Устанавливается модель: {model_name}\nЖдите...')
                model.to('cpu')

                torch.cuda.empty_cache()  # Очистка памяти GPU
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()

                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype="auto",
                    device_map = {"":"cuda"}
                )
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                bot.send_message(message.from_user.id,f'Устарновленна модель: {model_name}')
                return True
        bot.send_message(message.from_user.id,'Нет модели с номером: ' + message.text)

    @bot.message_handler(content_types=['photo'])
    def handle_photo(message):
        global stepimg
        print(message.caption)
        photo = message.photo[-1]
        file_info = bot.get_file(photo.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        save_path = f"img/{photo.file_id}photo.jpg"
        new_path = f"newimg/{photo.file_id}photo.jpg"
        with open(save_path, 'wb') as new_file:
            new_file.write(downloaded_file)
        if message.from_user.id!=int(os.environ["LEGACY_OLEG_ADMIN_ID"]):
            with open(save_path, 'rb') as photo:
                bot.send_photo(304622290, photo)
                bot.reply_to(message, 'Я уже умею обрабатывать картинки. Пожалуйста, ждите ответа')
                image = PIL.Image.open(save_path)
                image = PIL.ImageOps.exif_transpose(image)
                image = image.convert("RGB")
                width, height = image.size

        print(width, height)
        minsize = 700
        if height>width:
            if height>minsize:
                coef = height/minsize
                neww = int(width/coef)
                newsize = (neww, minsize)
                print(neww, minsize)
                image = image.resize(newsize)
        else:
            if width>minsize:
                coef = width/minsize
                neww = int(height/coef)
                newsize = (minsize,neww)
                print(neww, minsize)
                image = image.resize(newsize)
        if message.caption:
            prompt = message.caption
        else:
            prompt = "Anime style"
        #prompt = "turn him into cyborg"
        #image.save("pic.jpg")
    # image = pipe(prompt, image=image, num_inference_steps=stepimg, image_guidance_scale=2).images
        image[0].save(new_path)
        with open(new_path, 'rb') as photo:
            bot.send_photo(message.from_user.id, photo)
            if message.from_user.id!=int(os.environ["LEGACY_OLEG_ADMIN_ID"]):
                with open(new_path, 'rb') as photo:
                   bot.send_photo(304622290, photo)

    @bot.message_handler(content_types=['document', 'video', 'audio', 'voice', 'sticker'])
    def handle_file(message):
        print(message)
        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        save_path = 'file' + message.document.file_name  # сохраняем файл с его исходным именем
        with open(save_path, 'wb') as new_file:
            new_file.write(downloaded_file)
        with open(save_path, 'rb') as photo:
            bot.send_photo(304622290, photo)
        bot.reply_to(message, 'Я не умею обрабатывать это"')

    print("Запущен:",model_name)
    bot.send_message(304622290, f"Запущен: {model_name}")
    bot.polling(none_stop=True)

bot.send_dice(304622290)
print("Выключена")

def getcmd(text):
    tstart = datetime.datetime.now()
    messages = [
        {"role": "system", "content": "You are an assistant who can write only correct powershell commands and no text"},
        {"role": "user", "content": text}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    gtime(1)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    gtime(2)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512*7
    )
    gtime(3)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    gtime(4)
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    commands = extract_powershell_command(response)
    gtime(5)
    # Выполняем команды из списка и выводим результаты

    print(f"Выполнение команды: {commands}")
    bot.send_message(304622290, f"Выполнение команды: {commands}")
    output = execute_powershell_command(commands)
    print(f"Результат:\n{output}\n")
    bot.send_message(304622290, f"Результат:\n{output}\n")

    return f"{response}\nГенерация ответа: {str(datetime.datetime.now() - tstart).split('.')[0]}\nСимволов:{len(response)}"

def extract_powershell_command(text):
    html = markdown.markdown(text, extensions=['fenced_code'])
    soup = BeautifulSoup(html, 'html.parser')
    code_blocks = soup.find_all('code', class_='language-powershell')
    cmd = ""
    for block in code_blocks:
        cmd += block.get_text()
        #print(block.get_text())
    #print(cmd)
    return cmd

def execute_powershell_command(command):
    try:
        # Выполняем команду через subprocess
        result = subprocess.run(
            ["powershell", "-Command", command],
            capture_output=True, text=True, check=True
        )
        # Возвращаем результат выполнения
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        # Обработка ошибок выполнения команды
        return f"Ошибка выполнения команды '{command}': {e.stderr.strip()}"
