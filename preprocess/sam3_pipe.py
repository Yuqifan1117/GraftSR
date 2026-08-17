import os
import json
import numpy as np
import sam3
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import queue

sam3_root = os.path.dirname(sam3.__file__)

import torch

prompt_dict = {
    'tops': ['T恤', '马甲', '衬衫', '吊带背心', '针织衫', '背心', '针织开衫', '开衫', '上衣', '小衫', 'Polo衫', '长袖T恤', '风衣', '西装外套', '牛仔外套', '夹克', '外套', '长袖上衣', '打底衫', '皮夹克', '防晒衫', '风衣外套', '皮衣', '罩衫',
        '抹胸', '毛衣', '短外套', '运动内衣', '短袖衬衫', '卫衣', '冲锋衣', '两件套上衣', '衬衣', '针织马甲', '针织上衣', '牛仔夹克', '防晒服', '牛仔衬衫', '雪纺衫', '披肩', '卫衣外套', '针织罩衫', '针织背心', '吊带', '连帽卫衣', '套头衫', 
        '马甲背心', '吊带上衣', '衬衫外套', '防晒衣', '牛仔马甲', '开衫外套', '抹胸背心', '运动衫', '抹胸上衣', '吊带衫', '短袖上衣', '衬衫风衣', 'POLO衫', '连帽长袖T恤', '防晒外套', '毛针织衫', '针织短袖', '叠穿上衣', '无袖T恤', '大衣', '蕾丝衫', '针织T恤', 
        '连帽T恤', '西服', '无袖上衣', '无袖衬衫', '防晒上衣', '羽绒马甲', '连帽衫', '云肩', '衬衫背心', '开衫卫衣', '马夹', '夹克外套', '茄克', '卫衣开衫', '', '羽绒服', '棒球服夹克', '蕾丝衬衫', '胸衣', '短上衣', '披肩外套', '皮外套', '披衫', '鱼骨胸衣', 
        '运动外套', '连帽开衫', '西装夹克', '毛织上衣', '针织马甲背心', 'T恤裙', '文胸', '针织披肩', '短袖T恤', '牛仔风衣', '棒球夹克', 'POLO衫卫衣', '连帽衬衫', '背心吊带', '针织毛衣', '棒球服', '衬衫式外套', '针织小衫', '比甲', '短袖针织衫', '棉服', 
        '连帽外套', '短大衣', '马甲外套', '皮草外套', '斗篷', '针织外套', '风衣夹克', '娃娃衫', '皮衣外套', '水手服', 'T恤衫', '汉服上衣', '羊绒衫', '蕾丝上衣', '罩衫外套', '两件套T恤', '短风衣', '鹅绒服', '针织套衫', '针织毛衫', '连体上衣', '短袖衫', 
        '针织套头衫', '牛仔衬衫外套', '棉衣', '毛呢外套', '斗篷风衣', '哈林顿夹克', '长袖衫', '棉服外套', '羽绒棉服', '连帽卫衣开衫', '造型衫', '连帽针织衫', '软壳衣', '开衫毛衣', '皮风衣', '羊毛衫', '毛套衫', '斗篷披肩', '羽绒外套', '拉链衫', '披风', 
        '毛衣外套', '连帽毛衣', '挂脖上衣', '夹棉袄', '毛衣开衫', '拼接上衣', '斗篷毛衣', '毛呢夹克', '棉袄', '斗篷外套', '假两件上衣', '无袖衬衣', '派克外套', '针织打底衫', '斗篷披肩外套', '工装夹克', '连帽开衫卫衣', '棒球服外套', '大襟衫', '棒球外套', 
        '披肩上衣', '毛衫', '棉马甲', '羽绒夹克', '家居服上衣', '挂脖短上衣', '棉衣外套', '毛织衫', '短袄', '派克羽绒服', '运动上衣', '假领子', '皮草马甲', '坎肩上衣', '短袖套衫', '打底上衣', '毛绒外套', '毛马甲', '羽绒派克服', '风衣羽绒服', '夹棉衬衫',
        '外披', '情侣毛衣', '连帽上衣', '大披肩', '短衫', '毛衣裙', '亨利衫', '皮草背心', '蝙蝠衫', '毛衣背心', '披袄', '紧身衣', '棉服马甲', '抓绒衣', '棉服衬衫', '夹棉外套', '假领', '套衫', '飞行员夹克', '滑雪服', '皮草披肩', '羽绒风衣', '小外套',
        '针织卫衣', '西装风衣', '牛仔棉衣', '小上衣', '袄', '披肩斗篷', '球衣', '短夹克', '皮衬衫', '毛织开衫', '针织衬衫', '毛外套', '棉背心', '抓绒外套', '开衫吊带两件套', '连帽针织套衫', '派克棉服', '飞行夹克', '摇粒绒外套', '羽绒背心', '棉夹克',
        '圆领袍', '羽绒衬衫', '毛衣马甲', '抓绒开衫', '针织西装', 'T恤开衫', '连帽套衫', '棉外套', '围巾披肩', 'T恤套装', '衬衫夹克', '小西装外套', '西装马甲', '坎肩', 'V领针织衫', '防晒罩衫', '衬衫上衣', '假两件针织开衫', 'polo领针织衫', '假两件针织衫'],
    'bottoms': ['半身裙', '短裙', '卫裤', '休闲裤', '短裤', '半裙', '阔腿裤', '牛仔裤', '牛仔裙', '裤子', '长裤', '牛仔半裙', '中裤', '西裤', '裙裤', '牛仔短裤', '工装裤', '马面裙', '运动裤', '西装裤', '牛仔短裙', '五分裤', '铅笔裤', '牛仔半身裙', 
        '七分裤', '灯笼裤', '裤装', '萝卜裤', '裤裙', '九分裤', '直筒裤', '微喇裤', '短裙裤', '针织裤', '打底裤', '伞兵裤', '瑜伽裤', '阔腿长裤', '西装裙裤', '鲨鱼裤', '牛仔中裤', '喇叭牛仔裤', '哈伦裤', '喇叭裤', '百褶裙', '腰封屁帘', '微喇叭裤', 
        '半身裙裤', '短裤裙', '牛仔裙裤', '冲锋裤', '中裙', '腿套', '牛仔长裤', '皮裤', '半身裙配饰', '半身短裙', '皮裙', '长裙', '铅笔裙', '休闲长裤', '腰搭', '束脚裤', '皮短裤', '裙帘', '针织裙', '腰封', '西装短裤', '棉裤', '套裙', '外套裙', 
        '羽绒裤', '五分中裤', '西装长裤', '运动长裤', '针织长裙', '毛呢裤', '弯刀裤', '热裤', '阔腿牛仔裤', '运动卫裤', '五分阔腿裤', '中短裤', '腰饰', '阔腿短裤', '老爹裤', '七分靴裤', '直筒裙', '工装长裤', '睡裤', '屁帘', '迷彩七分裤', '西装喇叭裤'],
    'outfit': ['套装', '连衣裙', '两件套', '连衣裙套装', '背带裤', '连体衣', '吊带长裙', '汉服套装', '背带裙', '鱼尾裙', '围裙', '睡裙', '连体裤', '连体背心', '吊带裙', '三件套', '套装裙', '西装套装', '汉服', '运动套装', '旗袍', '针织套装', '西装', 
        '卫衣套装', '家居服套装', '情侣装', '旗袍裙', '衬衫裙', '卫衣连衣裙', '睡衣套装', '连体衬衣', '休闲套装', '罩裙', '针织开衫套装', '衬衫连衣裙', '卫衣裙', '针织长袍', '连体背带裤', '背带长裤', '连身裙', '连身裤', '针织两件套', '半身裙套装', 
        '阔腿裤套装', '上衣套装', '夹克套装', '西装套裤', '两件套连衣裙', '针织连衣裙', '连体裙', '旗袍套装', '衬衫套装', '制服套装', '睡衣', '背带连衣裙', '外套短裤套装', '连体短裤', '水手服套装', '马甲套装', '背心裙套装', '牛仔套装', '背心裙', '针织衫套装', 
        '短袖套装', '卫衣裙套装', '长袍', '休闲裤套装', '长衫', 'JK制服套装', '套装连衣裙', '吊带连衣裙', '针织马甲套装', '毛衣套装', '大衣套装', '毛呢大衣', '吊带套装', '马甲背心套装', '礼服', '礼服裙', '派克服', '西装套裙', '卫裙', '两件套套装', 
        '针织背心套装', '棉服套装', '背心连衣裙', '西装套组', '皮大衣', '两件套裙装', '斗篷大衣', '保暖内衣套装', '礼服套装', '连体睡衣', '家居服', '直裾袍', '呢大衣', '针织开衫背心套装', '连衣裙外套', '连衣裙两件套', '滑雪服套装', '皮草大衣', '舞蹈服', 
        '针织衫/短裙套装', '西装大衣', '两件套针织衫', '背带裙套装', '围脖手套套装', '派克大衣', '外套套装', '耳罩手套套装', '服装集合', '披帘罩裙', '连体滑雪服', '羽绒服套装', '连体衬衫', '连体打底衫', '连衣长袍', '配饰套装', '家居套装', '连衣裙/半身裙',
        '秀禾服', '马面裙套装', '披肩大衣', '时尚套装', '西服外套', '开衫套装', '围巾手套组合', '呢子大衣', '马甲连衣裙套装', '保暖内衣', '长袄', '大衣马甲套装', '帽子围巾手套套装', '卫衣裤', '马甲裙', '民族风套装', '职业套裙', '针织大衣', '两件套礼服',
        '针织开衫两件套', '背心套装', '衬衫半身裙套装', '褙子', 'T恤连衣裙', '睡袍', '上衣, 半身裙', '旗袍短裙', '裤装套装', '羽绒服, 休闲裤', '吊带长裙套装'],
    'shoes': ['高跟鞋', '单鞋', '乐福鞋', '短靴', '玛丽珍鞋', '凉鞋', '半拖鞋', '凉拖', '穆勒鞋', '休闲鞋', '运动鞋', '玛丽珍单鞋', '板鞋', '小白鞋', '老爹鞋', '凉拖鞋', '护士鞋', '平底鞋', '半拖', '渔夫鞋', '德训鞋', '帆布鞋', '芭蕾舞鞋', '女鞋', '凉靴', 
        '鞋子', '拖鞋', '松糕鞋', '厚底鞋', '皮鞋', '骑士靴', '长筒靴', '雪地靴', '马丁靴', '高筒靴', '靴子', '长靴', '中筒靴', '过膝靴', '勃肯鞋', '牛津鞋', '袜靴', '过膝长靴', '小皮鞋', '棉鞋', '分趾靴', '棉拖鞋', '运动靴', '人字拖', '跑鞋', '穆勒拖'],
    'socks': ['袜子', '长筒袜', '过膝袜', '袜套', '中筒袜', '堆堆袜'],
    'bag': ['腋下包', '水桶包', '双肩包', '波士顿包', '托特包', '手提包', '单肩包', '斜挎包', '法棍包', '马鞍包', '流浪包', '小方包', '化妆包', 'Hobo包', '盒子包', '保龄球包', '双肩背包', '公文包', '手机包', 'hobo包', '菜篮子包', '链条包', '帆布包', '包', 
        '凯莉包', '邮差包', '方包', '背包', '圆筒包', '枕头包', 'HOBO包', '挂脖包', '零钱包', '旅行包', '编织包', '钱包', '单肩斜挎包', '女包'],
    'hat': ['鸭舌帽', '遮阳帽', '渔夫帽', '棒球帽', '空顶帽', '草帽', '帽子', '冷帽', '报童帽', '贝雷帽', '针织帽', '堆堆帽', '礼帽', '八角帽', '毛线帽', '雷锋帽', '护耳帽', '盆帽', '针织冷帽'],
    'face mask': ['防晒面罩', '口罩', '面罩'],
    'glasses': ['墨镜', '太阳镜', '太阳眼镜', '眼镜'], 
    'drawstring': ['绳饰', '腰绳', '腰链'], 
    'scarf': ['丝巾', '围巾', '三角巾', '方巾', '长巾', '领巾', '长丝带', '小方巾', '围脖', '围巾套装', '皮草围巾'],
    'necklace': ['卡包项链', '项链', '挂链'],
    'hair accessories': ['抓夹', '发箍', '发夹', '发圈', '发绳', '发带'],
    'belt': ['腰带', '皮带'],
    'ring': ['戒指'],
    'gloves': ['半指手套', '手套'],
    'tie': ['领带', '领带夹'],
    'collar': ['项圈'],
    'keychain': ['钥匙扣'],
    'brooch': ['胸针'],
    'bracelet': ['手链'],
    'pantyhose': ['连裤袜', '连裤丝袜'],
    'ID card holder': ['证件套', '卡套', '卡套挂链'],
    'ear rings': ['耳钉', '耳环', '耳坠', '耳骨夹'],
    'ear muffs': ['耳罩', '耳套', '护耳罩'],
    'balaclava': ['巴拉克拉法帽', '头套', '包头帽'],
    'pendant': ['挂件', '挂饰'],
    'hooded scarf': ['连帽围巾', '帽子围巾', '耳罩围脖', '帽子围巾套装', '连帽针织围脖', '帽围', '围巾冷帽', '围巾帽', '一体帽'],
    'sleep mask': ['眼罩'],
    'strap': ['肩带'],
    'arm warmers': ['防晒袖套']
}

ori_txt = '/data/oss_bucket_0/Users/yuqifan/ODTSR+/benchmark/paper_results/ref_ori.txt'
prompt_txt = '/data/oss_bucket_0/Users/yuqifan/ODTSR+/benchmark/paper_results/prompt_ori.txt'
with open(ori_txt, 'r', encoding='utf-8') as f:
    ori_list = f.read().splitlines()
print(len(ori_list))

with open(prompt_txt, 'r', encoding='utf-8') as f:
    prompt_list = f.read().splitlines()
print(len(prompt_list))

translate_dict = dict()
for txt_path in prompt_list:
    with open(txt_path, 'r', encoding='utf-8') as f:
        prompt = f.read().splitlines()[0]
        item = prompt.split('。 ')[-1].split('：')[0]
        exist_word = False
        for k in prompt_dict:
            if item in prompt_dict[k]:
                translate_dict[item] = k
                exist_word = True
                break
        if not exist_word:
            for k in prompt_dict:
                for it in prompt_dict[k]:
                    if it in item:
                        translate_dict[item] = k
                        exist_word = True
                        break
                    if exist_word:
                        break
        if not exist_word:
            print(item)
print(translate_dict)
# json.dump(translate_dict, open('/data/oss_bucket_0/Users/yuqifan/ODTSR+/benchmark/realworld/translate_items.json', 'w'))

# turn on tfloat32 for Ampere GPUs
# https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# use bfloat16 for the entire notebook
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

def save_mask_async(mask_uint8, dest_pth, size):
    """异步保存掩码图片"""
    try:
        img = Image.fromarray(mask_uint8, mode='L')
        img.save(dest_pth)
    except Exception as e:
        print(f"Error saving {dest_pth}: {e}")

bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
model = build_sam3_image_model(bpe_path=bpe_path, checkpoint_path='/mnt/workspace/heye/ckpt/facebook/sam3/sam3.pt')
processor = Sam3Processor(model, confidence_threshold=0.5)

dest_root = '/data/oss_bucket_0/Users/yuqifan/ODTSR+/benchmark/paper_results/ref_ori_mask_sam3'
dest_txt_root = '/data/oss_bucket_0/Users/yuqifan/ODTSR+/benchmark/paper_results/ref_crop_mask_sam3.txt'
os.makedirs(dest_root, exist_ok=True)
new_data = []


count0, count1 = 0, 0
save_futures = []
MAX_PENDING_SAVES = 8  # 最大待保存任务数，避免内存占用过多

# 使用线程池进行异步保存
executor = ThreadPoolExecutor(max_workers=4)

pbar = tqdm(ori_list)
for idx, ori_pth in enumerate(pbar):
    ori_pth = ori_pth.strip()
    base_name = os.path.basename(ori_pth)
    dest_pth = os.path.join(dest_root, base_name.split('.')[0] + '.png')
    new_data.append(dest_pth + '\n')
    
    # 如果待保存任务过多，等待部分完成
    if len(save_futures) >= MAX_PENDING_SAVES:
        # 等待最早的一个任务完成
        save_futures.pop(0).result()
    
    txt_pth = prompt_list[idx]
    with open(txt_path, 'r', encoding='utf-8') as f:
        prompt = f.read().splitlines()[0]
        item = prompt.split('。 ')[-1].split('：')[0]
    
    p = translate_dict[item]

    image = Image.open(ori_pth).convert('RGB')
    inference_state = processor.set_image(image)
    processor.reset_all_prompts(inference_state)
    inference_state = processor.set_text_prompt(state=inference_state, prompt=p)
    
    if inference_state['masks'].shape[0] == 0:
        width, height = image.size
        mask_uint8 = np.ones((height, width), dtype=np.uint8) * 255
        count0 += 1
    else:
        mask_2d = torch.any(inference_state['masks'], dim=0).squeeze(0)
        mask_uint8 = (mask_2d * 255).to(torch.uint8).cpu().numpy()
        count1 += 1
    
    # 异步保存，不阻塞主线程
    future = executor.submit(save_mask_async, mask_uint8, dest_pth, image.size)
    save_futures.append(future)
    
    pbar.set_postfix(count0=count0, count1=count1, pending_saves=len(save_futures))

# 等待所有保存任务完成
for future in save_futures:
    future.result()

executor.shutdown(wait=True)

with open(dest_txt_root, 'w', encoding='utf-8') as file:
    file.writelines(new_data)