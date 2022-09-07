from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display


def int_to_str(n):
    n_str = str(n)
    for i in range(len(n_str)-3, 0, -3):
        n_str = n_str[0:i] + ' , ' + n_str[i:]
    return n_str


if __name__ == '__main__':
    lst = []
    file = open('list.txt', 'r', encoding='utf-8').read().split('\n')
    for line in file:
        lst.append(line.split('.'))

    start_pos_x = 700
    start_pos_y = 510
    row_height = 82
    total = 0
    color = (255, 0, 0)
    font = ImageFont.truetype('yekan.ttf', size=40)
    img = Image.open('raw.jpg')
    draw = ImageDraw.Draw(img)
    for i in range(len(lst)):
        draw.text((start_pos_x, ((i + 1) * row_height) + start_pos_y), get_display(arabic_reshaper.reshape(lst[i][0])),
                  fill=color, font=font)
        draw.text((start_pos_x - 120, ((i + 1) * row_height) + start_pos_y),
                  get_display(arabic_reshaper.reshape(lst[i][1])),
                  fill=color, font=font)
        draw.text((start_pos_x - 310, ((i + 1) * row_height) + start_pos_y),
                  get_display(arabic_reshaper.reshape(lst[i][2])),
                  fill=color, font=font)
        price = int(lst[i][1]) * int(lst[i][2])
        total += price
        draw.text((start_pos_x - 620, ((i + 1) * row_height) + start_pos_y),
                  get_display(arabic_reshaper.reshape(int_to_str(price))),
                  fill=color, font=font)

    draw.text((start_pos_x - 620, 1520), get_display(arabic_reshaper.reshape(int_to_str(total))),
              fill=color, font=font)

    img.save('factor.jpeg')
    print('...done.')
