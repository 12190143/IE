import re
import chardet
from datetime import datetime, timedelta
import re
from datetime import datetime, timedelta
from dateutil.parser import parse
import jieba.posseg as psg

# 匹配正则表达式
matchs = {
    1: (r'\d{4}%s\d{1,2}%s\d{1,2}%s \d{1,2}%s\d{1,2}%s\d{1,2}%s', '%%Y%s%%m%s%%d%s %%H%s%%M%s%%S%s'),
    2: (r'\d{4}%s\d{1,2}%s\d{1,2}%s \d{1,2}%s\d{1,2}%s', '%%Y%s%%m%s%%d%s %%H%s%%M%s'),
    3: (r'\d{4}%s\d{1,2}%s\d{1,2}%s', '%%Y%s%%m%s%%d%s'),
    4: (r'\d{2}%s\d{1,2}%s\d{1,2}%s', '%%y%s%%m%s%%d%s'),

    # 没有年份
    5: (r'\d{1,2}%s\d{1,2}%s \d{1,2}%s\d{1,2}%s\d{1,2}%s', '%%m%s%%d%s %%H%s%%M%s%%S%s'),
    6: (r'\d{1,2}%s\d{1,2}%s \d{1,2}%s\d{1,2}%s', '%%m%s%%d%s %%H%s%%M%s'),
    7: (r'\d{1,2}%s\d{1,2}%s', '%%m%s%%d%s'),

    # 没有年月日
    8: (r'\d{1,2}%s\d{1,2}%s\d{1,2}%s', '%%H%s%%M%s%%S%s'),
    9: (r'\d{1,2}%s\d{1,2}%s', '%%H%s%%M%s'),
}

# 正则中的%s分割
splits = [
    {1: [('年', '月', '日', '点', '分', '秒'), ('-', '-', '', ':', ':', ''), ('\/', '\/', '', ':', ':', ''),
         ('\.', '\.', '', ':', ':', '')]},
    {2: [('年', '月', '日', '点', '分'), ('-', '-', '', ':', ''), ('\/', '\/', '', ':', ''), ('\.', '\.', '', ':', '')]},
    {3: [('年', '月', '日'), ('-', '-', ''), ('\/', '\/', ''), ('\.', '\.', '')]},
    {4: [('年', '月', '日'), ('-', '-', ''), ('\/', '\/', ''), ('\.', '\.', '')]},

    {5: [('月', '日', '点', '分', '秒'), ('-', '', ':', ':', ''), ('\/', '', ':', ':', ''), ('\.', '', ':', ':', '')]},
    {6: [('月', '日', '点', '分'), ('-', '', ':', ''), ('\/', '', ':', ''), ('\.', '', ':', '')]},
    {7: [('月', '日'), ('-', ''), ('\/', ''), ('\.', '')]},

    {8: [('点', '分', '秒'), (':', ':', '')]},
    {9: [('点', '分'), (':', '')]},
]


def func(parten, tp):
    re.search(parten, parten)


parten_other = '\d+天前|\d+分钟前|\d+小时前|\d+秒前'


class TimeFinder(object):

    def __init__(self, base_date=None):
        self.base_date = base_date
        self.match_item = []

        self.init_args()
        self.init_match_item()

    def init_args(self):
        # 格式化基础时间
        if not self.base_date:
            self.base_date = datetime.now()
        if self.base_date and not isinstance(self.base_date, datetime):
            try:
                self.base_date = datetime.strptime(self.base_date, '%Y-%m-%d %H:%M:%S')
            except Exception as e:
                raise 'type of base_date must be str of%Y-%m-%d %H:%M:%S or datetime'

    def init_match_item(self):
        # 构建穷举正则匹配公式 及提取的字符串转datetime格式映射
        for item in splits:
            for num, value in item.items():
                match = matchs[num]
                for sp in value:
                    tmp = []
                    for m in match:
                        tmp.append(m % sp)
                    self.match_item.append(tuple(tmp))

    def get_time_other(self, text):
        m = re.search('\d+', text)
        if not m:
            return None
        num = int(m.group())
        if '天' in text:
            return self.base_date - timedelta(days=num)
        elif '小时' in text:
            return self.base_date - timedelta(hours=num)
        elif '分钟' in text:
            return self.base_date - timedelta(minutes=num)
        elif '秒' in text:
            return self.base_date - timedelta(seconds=num)

        return None

    def find_time(self, text):
        # 格式化text为str类型
        if isinstance(text, bytes):
            encoding = chardet.detect(text)['encoding']
            text = text.decode(encoding)

        res = []
        parten = '|'.join([x[0] for x in self.match_item])

        parten = parten + '|' + parten_other
        match_list = re.findall(parten, text)
        if not match_list:
            return None
        for match in match_list:
            for item in self.match_item:
                try:
                    date = datetime.strptime(match, item[1].replace('\\', ''))
                    if date.year == 1900:
                        date = date.replace(year=self.base_date.year)
                        if date.month == 1:
                            date = date.replace(month=self.base_date.month)
                            if date.day == 1:
                                date = date.replace(day=self.base_date.day)
                    res.append(datetime.strftime(date, '%Y-%m-%d %H:%M:%S'))
                    break
                except Exception as e:
                    date = self.get_time_other(match)
                    if date:
                        res.append(datetime.strftime(date, '%Y-%m-%d %H:%M:%S'))
                        break
        if not res:
            return None
        return res


def check_time_valid(word):
    # 对提取的拼接日期串进行进一步处理，以进行有效性判断
    m = re.match("\d+$", word)
    if m:
        if len(word) <= 6:
            return None
    word1 = re.sub('[号|日]\d+$', '日', word)
    if word1 != word:
        return check_time_valid(word1)
    else:
        return word1


UTIL_CN_NUM = {'零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
'五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
'0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
'5': 5, '6': 6, '7': 7, '8': 8, '9': 9
}
UTIL_CN_UNIT = {'十': 10, '百': 100, '千': 1000, '万': 10000}


def cn2dig(src):
    if src == "":
        return None
    m = re.match("\d+", src)
    if m:
        return int(m.group(0))
    rsl = 0
    unit = 1
    for item in src[::-1]:
        if item in UTIL_CN_UNIT.keys():
            unit = UTIL_CN_UNIT[item]
        elif item in UTIL_CN_NUM.keys():
            num = UTIL_CN_NUM[item]
            rsl += num * unit
        else:
            return None
    if rsl < unit:
        rsl += unit
    return rsl


def year2dig(year):
    res = ''
    for item in year:
        if item in UTIL_CN_NUM.keys():
            res = res + str(UTIL_CN_NUM[item])
        else:
            res = res + item
    m = re.match("\d+", res)
    if m:
        if len(m.group(0)) == 2:
            return int(datetime.datetime.today().year/100)*100 + int(m.group(0))
        else:
            return int(m.group(0))
    else:
        return None


def parse_datetime(msg):
    # 将每个提取到的文本日期串进行时间转换。
    # print("parse_datetime开始处理：",msg)
    if msg is None or len(msg) == 0:
        return None
    try:
        msg = re.sub("年"," ",msg)# parse不认识"年"字
        dt = parse(msg, yearfirst=True, fuzzy=True)
        # print(dt)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        m = re.match(r"([0-9零一二两三四五六七八九十]+年)?([0-9一二两三四五六七八九十]+月)?([0-9一二两三四五六七八九十]+[号日])?([上中下午晚早]+)?([0-9零一二两三四五六七八九十百]+[点:\.时])?([0-9零一二三四五六七八九十百]+分?)?([0-9零一二三四五六七八九十百]+秒)?",
                    msg)
        if m.group(0) is not None:
            res = {
                "year": m.group(1),
                "month": m.group(2),
                "day": m.group(3),
                "hour": m.group(5) if m.group(5) is not None else '00',
                "minute": m.group(6) if m.group(6) is not None else '00',
                "second": m.group(7) if m.group(7) is not None else '00',
            }
            params = {}

            for name in res:
                if res[name] is not None and len(res[name]) != 0:
                    tmp = None
                    if name == 'year':
                        tmp = year2dig(res[name][:-1])
                    else:
                        tmp = cn2dig(res[name][:-1])
                    if tmp is not None:
                        params[name] = int(tmp)
            target_date = datetime.today().replace(**params)
            is_pm = m.group(4)
            if is_pm is not None:
                if is_pm == u'下午' or is_pm == u'晚上' or is_pm =='中午':
                    hour = target_date.time().hour
                    if hour < 12:
                        target_date = target_date.replace(hour=hour + 12)
            return target_date.strftime('%Y-%m-%d %H:%M:%S')
        else:
            return None


def time_extract(text):
    time_res = []
    word = ''
    keyDate = {'今天': 0, '至今': 0, '明天':1, '后天': 2, '大后天': 3}
    for k, v in psg.cut(text):
        if k in keyDate:
            if word != '':
                time_res.append(word)
            word = (datetime.today() + timedelta(days=keyDate.get(k, 0))).strftime('%Y年%m月%d日')
        elif word != '':
            if v in ['m', 't']:
                word = word + k
            else:
                time_res.append(word)
                word = ''
        elif v in ['m', 't']:
            word = k
    if word != '':
        time_res.append(word)
    # print('22222222',time_res)
    result = list(filter(lambda x: x is not None, [check_time_valid(w) for w in time_res]))
    final_res = [parse_datetime(w) for w in result]
    return [x for x in final_res if x is not None]


def test():
    timefinder = TimeFinder(base_date='2020-04-23 00:00:00')
    for text in ['2012年12月12日', '3小时前', '在2012/12/13哈哈', '时间2012-12-11 12:22:30', '日期2012-13-11', '测试2013.12.24',
                 '今天12:13', '明天12点', '下周']:
        res = timefinder.find_time(text)
        print('text----', text)
        print('res---', res)
        print(time_extract(text))

    print(time_extract("从2016年3月5日至今"))
    print(time_extract("在20160824-20180529的全部交易。"))
    print(time_extract("2017.6.12-7.10交！"))


if __name__ == '__main__':
    test()
