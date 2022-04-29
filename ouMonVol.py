# program for simulating flights cancellation, Monte-Carlo method
# Version Final-20220429
# Author: Kirk0830
# Github: https://github.com/kirk0830/
# Github pages: https://kirk0830.github.io/
# Prerequisition: python, py-numpy, py-matplotlib
# =======================================
# 1. Simulation parameters（模拟基本参数）
# AVE：阳性病例数量的平均数，用于随机生成后续航班阳性病例数量的参数
# DEV：阳性病例数量的标准差，目前仅用于高斯分布
# DAMP：当此参数设置为大于1的值，则平均确诊数量逐周上升，反之下降
# NITER：迭代次数，至少1000才可以得到比较稳定和可靠的结果
# SPAN：最大预测周数
# rand_dist：随机变量分布，支持norm, poisson, uni
AVE = 5
DEV = 2
DAMP = 1.0
NITER = 10000
SPAN = 12
rand_dist = 'norm'
# =======================================
# 2. Analysis tasks（模拟分析任务）
# L_CORR：航班相关性分析
# L_SENS：平均值参数局部敏感性分析
# L_TKTS：如果手中已经有一定数量的机票，可以打开这一标签，并在数组tks中指定周数
#         例如若持有第一周，第三周机票，则tks = [0, 2]
L_CORR = False
L_SENS = True
L_TKTS = True
# =======================================
# 3. Preconditions（模拟条件初始化）
# tks：已经购买的航班机票序号，下例为第七、八、九、十周
# csprecdx：给定一些周已经公布的确诊人数，如果两周并不连续，之间的周用-1占位，如
#           第一、三周分别是5, 7，则设置csprecdx = [5, -1, 7]
# week1, 2, 3：设置熔断后前三班稳飞
# 当前预设条件：南航
# 1. 确诊人数，4.14确诊人数6，4.21确诊人数7
# 2. 购票，第八周6.9，第十周6.23
# 3. 起飞初始条件，第一周、第二周、第三周即4.14，4.21，4.28均起飞
tks = [9, 11]
csprecdx = [6, 7]
week1 = True
week2 = True
week3 = True
# =======================================
# 4. Output（输出控制）
# GRAPH：利用matplotlib扩展包进行绘图，注意，如果L_CORR为True，则此标签只能控制
#        起飞概率的画图与否
# SAVE_FILE：是否将结果输出为txt文档
GRAPH = False
SAVE_FILE = False
# =======================================

# DEVELOPER READABLE SECTION BELOW...除非是debug不然下面的不用再看了
# --------------------------------------------------------------
_boollst0 = [1] * SPAN
_boollst0[0] = int(week1)
_boollst0[1] = int(week2)
_boollst0[2] = int(week3)

csprecdx = csprecdx+[-1]*(SPAN-len(csprecdx))

def do_fuse_policy_0(var_bool_fly, week, _fsnum, _fsmode = 'normal'):
    '''
    # do_fuse_policy_0\n
    Input parameters\n
    var_bool_fly: bool list in present iteration, contains all states of flights, [1, 0, 0, ...],\n
    week: present week that triggers fuse\n
    _fsnum: number of flights will be cancelled\n
    _fsmode: use for 4+4 case, otherwise leave it as normal as default\n
    In normal mode, only filghts after 3 weeks will be considered to cancel, while in other mode
    flights will be cancelled immediately, i.e., week1 and 2 both have more than 10 cases, flights
    will be cancelled from week3 rather than week4.
    '''
    _fscount = 0
    if _fsmode == 'normal':
        _disp = 3
    else:
        _disp = 1
        _fsnum = 4
    while ((week + _disp) < len(var_bool_fly)) and (_fscount < _fsnum):
        if var_bool_fly[week+_disp]:
            # start here!
            var_bool_fly[week+_disp] = 0
            _disp += 1
            _fscount += 1
        else:
            _disp += 1

    return var_bool_fly
from numpy import random as rd
def rd_case(ave = AVE, dist = 'norm'):

    if dist == 'norm':
        return int(rd.normal(ave, DEV))
    elif dist == 'uni':
        return int(ave*rd.uniform())
    elif dist == 'poisson':
        return rd.poisson(ave)
    else:
        return ave
from copy import deepcopy as dcp
def evol(rdave = AVE, nw = SPAN, ncyc = NITER):

    p = [0.] * nw
    p_corrmat = [[0. for i in range(nw)] for j in range(nw)]
    p_tickets = 0.
    
    for idx_iter in range(ncyc):
        
        _boollst = dcp(_boollst0)
        _csslog = [0] * nw
        _temp_bool_tks = 1
        for iw in range(nw):
            rdave *= DAMP
            if _boollst[iw]:
                _cs = rd_case(ave = rdave, dist = rand_dist)
                if csprecdx[iw] >= 0: 
                    _cs = csprecdx[iw]
                if _cs < 0: _cs = 0
                _csslog[iw] = _cs
                if _cs < 5:
                    pass
                elif _cs < 9:
                    _boollst = do_fuse_policy_0(
                        _boollst, 
                        week = iw,
                        _fsnum = 2
                        )
                elif _cs < 30:
                    if _csslog[iw-1] > 9:
                        _boollst = do_fuse_policy_0(
                            _boollst, 
                            week = iw,
                            _fsnum = 4,
                            _fsmode = 'sad'
                            )
                    else:
                        _boollst = do_fuse_policy_0(
                            _boollst, 
                            week = iw,
                            _fsnum = 4
                            )
                else:
                    _boollst = do_fuse_policy_0(
                        _boollst, 
                        week = iw,
                        _fsnum = 4,
                        _fsmode = 'sad'
                        )
            else:
                _csslog[iw] = 0
        for iw in range(nw):
            p[iw] += _boollst[iw]/ncyc
            if L_CORR:
                for jw in range(iw,nw):
                    if iw != jw:
                        p_corrmat[iw][jw] += (_boollst[iw] - 0.5)/(_boollst[jw] - 0.5)/ncyc
                        p_corrmat[jw][iw] = p_corrmat[iw][jw]
                    else:
                        p_corrmat[iw][jw] += 1/ncyc
            if L_TKTS:
                if tks.count(iw):
                    _temp_bool_tks *= _boollst[iw]+1
        if L_TKTS:
            if _temp_bool_tks > 1:
                p_tickets += 1/ncyc

    p = [round(ip, ndigits=4) for ip in p]
    return p, p_corrmat, p_tickets

def sens(entity1, entity2, pertur, dm = 0):

    # (entity1 - entity2)/pertur
    if dm == 0:
        return round((entity1 - entity2)/pertur, ndigits=4)
    elif dm == 1:
        return [round((entity1[i] - entity2[i])/pertur, ndigits=4) for i in range(len(entity1))]
    elif dm == 2:
        return [[round((entity1[i][j] - entity2[i][j])/pertur, ndigits=4) for i in range(len(entity1))] for j in range(len(entity1))]
import matplotlib.pyplot as plt
def graph_out(data, dm, title, color):

    if dm==2:
        plt.figure()
        plt.imshow(data, origin='lower',cmap=color)
        plt.colorbar()
        plt.title(title)
    elif dm==1:
        _lweek = ['week '+str(iweek) for iweek in range(SPAN)]
        plt.figure(figsize=(10,4))
        plt.bar(x = _lweek, height = data, color=color)
        plt.title(title)
    elif dm < 0:
        plt.show()

def main():

    p0, p0_corrmat, p0_tickets = evol()
    if L_SENS:
        pR, pR_corrmat, pR_tickets = evol(rdave=AVE+0.25)
        pL, pL_corrmat, pL_tickets = evol(rdave=AVE-0.25)
        s_p = [
            sens(entity1=pR, entity2=p0, pertur=0.25, dm=1),
            sens(entity1=p0, entity2=pL, pertur=0.25, dm=1)
        ]
        if L_CORR:
            s_corr = [
                sens(entity1=pR_corrmat, entity2=p0_corrmat, pertur=0.25, dm=2),
                sens(entity1=p0_corrmat, entity2=pL_corrmat, pertur=0.25, dm=2)
                ]
            graph_out(p0_corrmat, 2, 'Correlation analysis between flights in SPAN range', 'RdGy')
            graph_out(s_corr[0], 2, 'Correlation sensitivity analysis, RIGHT derivatives', 'RdGy')
            graph_out(s_corr[1], 2, 'Correlation sensitivity analysis, LEFT derivatives', 'RdGy')

            if L_TKTS:
                s_tickets = [
                    sens(entity1=pR_tickets, entity2=p0_tickets, pertur=0.25, dm=0),
                    sens(entity1=p0_tickets, entity2=pL_tickets, pertur=0.25, dm=0)
                ]
                print('Probabilities of successfully taking off with all tickets in hands: '+str(p0_tickets))
                print('Derivative right: '+str(s_tickets[0]))
                print('Derivative left:  '+str(s_tickets[1]))
                if GRAPH:
                    title0 = 'Probabilities for flights setting off, with average positive cases: '+str(AVE)+', DAMP = '+str(DAMP)
                    graph_out(p0, 1, title0, color='blue')
                    title1 = 'Probabilities sensitivity analysis, RIGHT derivatives'
                    title2 = 'Probabilities sensitivity analysis, LEFT derivatives'
                    graph_out(s_p[0], 1, title1, color='orange')
                    graph_out(s_p[1], 1, title2, color='orange')
                if SAVE_FILE:
                    from time import asctime
                    with open(file = asctime().replace(' ','-').replace(':','')+'.txt', mode = 'a+', encoding = 'utf-8') as f:
                        f.writelines('SECTION| Probabilities of flights taking off, START HERE\n')
                        f.writelines('P | Sensitivity_Right | Sensitivity_Left\n')
                        for idx_p0 in range(SPAN):
                            f.writelines('week '+str(idx_p0)+': '+str(p0[idx_p0])+' '+str(s_p[0][idx_p0])+' '+str(s_p[1][idx_p0])+'\n')
                        f.writelines('SECTION| Correlation of flights taking off, START HERE\n')
                        for idx_p_corrmat in range(SPAN):
                            strwrite = ''
                            for jdx_p_corrmat in range(SPAN):
                                strwrite += str(p0_corrmat[idx_p_corrmat][jdx_p_corrmat])+' '
                            f.writelines(strwrite+'\n')
                        f.writelines('SECTION| Correlation sensitivities of flights taking off, START HERE\n')
                        f.writelines('RIGHT\n')
                        for idx_p_corrmat in range(SPAN):
                            strwrite = ''
                            for jdx_p_corrmat in range(SPAN):
                                strwrite += str(s_corr[0][idx_p_corrmat][jdx_p_corrmat])+' '
                            f.writelines(strwrite+'\n')
                        f.writelines('LEFT\n')
                        for idx_p_corrmat in range(SPAN):
                            strwrite = ''
                            for jdx_p_corrmat in range(SPAN):
                                strwrite += str(s_corr[1][idx_p_corrmat][jdx_p_corrmat])+' '
                            f.writelines(strwrite+'\n')
                        f.writelines('SECTION| Probabilities and derivatives of successfully taking off with all tickets in hands\n')
                        f.writelines(str(p0_tickets)+' '+str(s_tickets[0])+' '+str(s_tickets[1])+'\n')
                graph_out([],-1,'','')
                return p0, p0_corrmat, p0_tickets, s_p, s_corr, s_tickets
            else:
                if GRAPH:
                    title0 = 'Probabilities for flights setting off, with average positive cases: '+str(AVE)+', DAMP = '+str(DAMP)
                    graph_out(p0, 1, title0, color='blue')
                    title1 = 'Probabilities sensitivity analysis, RIGHT derivatives'
                    title2 = 'Probabilities sensitivity analysis, LEFT derivatives'
                    graph_out(s_p[0], 1, title1, color='orange')
                    graph_out(s_p[1], 1, title2, color='orange')
                if SAVE_FILE:
                    from time import asctime
                    with open(file = asctime().replace(' ','-').replace(':','')+'.txt', mode = 'a+', encoding = 'utf-8') as f:
                        f.writelines('SECTION| Probabilities of flights taking off, START HERE\n')
                        f.writelines('P | Sensitivity_Right | Sensitivity_Left\n')
                        for idx_p0 in range(SPAN):
                            f.writelines('week '+str(idx_p0)+': '+str(p0[idx_p0])+' '+str(s_p[0][idx_p0])+' '+str(s_p[1][idx_p0])+'\n')
                        f.writelines('SECTION| Correlation of flights taking off, START HERE\n')
                        for idx_p_corrmat in range(SPAN):
                            strwrite = ''
                            for jdx_p_corrmat in range(SPAN):
                                strwrite += str(p0_corrmat[idx_p_corrmat][jdx_p_corrmat])+' '
                            f.writelines(strwrite+'\n')
                        f.writelines('SECTION| Correlation sensitivities of flights taking off, START HERE\n')
                        f.writelines('RIGHT\n')
                        for idx_p_corrmat in range(SPAN):
                            strwrite = ''
                            for jdx_p_corrmat in range(SPAN):
                                strwrite += str(s_corr[0][idx_p_corrmat][jdx_p_corrmat])+' '
                            f.writelines(strwrite+'\n')
                        f.writelines('LEFT\n')
                        for idx_p_corrmat in range(SPAN):
                            strwrite = ''
                            for jdx_p_corrmat in range(SPAN):
                                strwrite += str(s_corr[1][idx_p_corrmat][jdx_p_corrmat])+' '
                            f.writelines(strwrite+'\n')
                graph_out([],-1,'','')
                return p0, p0_corrmat, s_p, s_corr
        else:
            if L_TKTS:
                s_tickets = [
                    sens(entity1=pR_tickets, entity2=p0_tickets, pertur=0.25, dm=0),
                    sens(entity1=p0_tickets, entity2=pL_tickets, pertur=0.25, dm=0)
                ]
                print('Probabilities of successfully taking off with all tickets in hands: '+str(p0_tickets))
                print('Derivative right: '+str(s_tickets[0]))
                print('Derivative left:  '+str(s_tickets[1]))
                if GRAPH:
                    title0 = 'Probabilities for flights setting off, with average positive cases: '+str(AVE)+', DAMP = '+str(DAMP)
                    graph_out(p0, 1, title0, color='blue')
                    title1 = 'Probabilities sensitivity analysis, RIGHT derivatives'
                    title2 = 'Probabilities sensitivity analysis, LEFT derivatives'
                    graph_out(s_p[0], 1, title1, color='orange')
                    graph_out(s_p[1], 1, title2, color='orange')
                    graph_out([],-1,'','')
                if SAVE_FILE:
                    from time import asctime
                    with open(file = asctime().replace(' ','-').replace(':','')+'.txt', mode = 'a+', encoding = 'utf-8') as f:
                        f.writelines('SECTION| Probabilities of flights taking off, START HERE\n')
                        f.writelines('P | Sensitivity_Right | Sensitivity_Left\n')
                        for idx_p0 in range(SPAN):
                            f.writelines('week '+str(idx_p0)+': '+str(p0[idx_p0])+' '+str(s_p[0][idx_p0])+' '+str(s_p[1][idx_p0])+'\n')
                        f.writelines('SECTION| Probabilities and derivatives of successfully taking off with all tickets in hands\n')
                        f.writelines(str(p0_tickets)+' '+str(s_tickets[0])+' '+str(s_tickets[1])+'\n')
                return p0, p0_tickets, s_p, s_tickets
            else:
                if GRAPH:
                    title0 = 'Probabilities for flights setting off, with average positive cases: '+str(AVE)+', DAMP = '+str(DAMP)
                    graph_out(p0, 1, title0, color='blue')
                    title1 = 'Probabilities sensitivity analysis, RIGHT derivatives'
                    title2 = 'Probabilities sensitivity analysis, LEFT derivatives'
                    graph_out(s_p[0], 1, title1, color='orange')
                    graph_out(s_p[1], 1, title2, color='orange')
                    graph_out([],-1,'','')
                if SAVE_FILE:
                    from time import asctime
                    with open(file = asctime().replace(' ','-').replace(':','')+'.txt', mode = 'a+', encoding = 'utf-8') as f:
                        f.writelines('SECTION| Probabilities of flights taking off, START HERE\n')
                        f.writelines('P | Sensitivity_Right | Sensitivity_Left\n')
                        for idx_p0 in range(SPAN):
                            f.writelines('week '+str(idx_p0)+': '+str(p0[idx_p0])+' '+str(s_p[0][idx_p0])+' '+str(s_p[1][idx_p0])+'\n')
                return p0, s_p
    else:
        if L_CORR:
            graph_out(p0_corrmat, 2, 'Correlation analysis between flights in SPAN range', 'RdGy')
            if L_TKTS:
                print('Probabilities of successfully taking off with all tickets in hands: '+str(p0_tickets))
                if GRAPH:
                    title0 = 'Probabilities for flights setting off, with average positive cases: '+str(AVE)+', DAMP = '+str(DAMP)
                    graph_out(p0, 1, title0, color='blue')
                if SAVE_FILE:
                    from time import asctime
                    with open(file = asctime().replace(' ','-').replace(':','')+'.txt', mode = 'a+', encoding = 'utf-8') as f:
                        f.writelines('SECTION| Probabilities of flights taking off, START HERE\n')
                        for idx_p0 in range(SPAN):
                            f.writelines('week '+str(idx_p0)+': '+str(p0[idx_p0])+'\n')
                        f.writelines('SECTION| Correlation of flights taking off, START HERE\n')
                        for idx_p_corrmat in range(SPAN):
                            strwrite = ''
                            for jdx_p_corrmat in range(SPAN):
                                strwrite += str(p0_corrmat[idx_p_corrmat][jdx_p_corrmat])+' '
                            f.writelines(strwrite+'\n')
                        f.writelines('SECTION| Probabilities of successfully taking off with all tickets in hands\n')
                        f.writelines(str(p0_tickets)+'\n')
                graph_out([],-1,'','')
                return p0, p0_corrmat, p0_tickets
            else:
                if GRAPH:
                    title0 = 'Probabilities for flights setting off, with average positive cases: '+str(AVE)+', DAMP = '+str(DAMP)
                    graph_out(p0, 1, title0, color='blue')
                if SAVE_FILE:
                    from time import asctime
                    with open(file = asctime().replace(' ','-').replace(':','')+'.txt', mode = 'a+', encoding = 'utf-8') as f:
                        f.writelines('SECTION| Probabilities of flights taking off, START HERE\n')
                        for idx_p0 in range(SPAN):
                            f.writelines('week '+str(idx_p0)+': '+str(p0[idx_p0])+'\n')
                        f.writelines('SECTION| Correlation of flights taking off, START HERE\n')
                        for idx_p_corrmat in range(SPAN):
                            strwrite = ''
                            for jdx_p_corrmat in range(SPAN):
                                strwrite += str(p0_corrmat[idx_p_corrmat][jdx_p_corrmat])+' '
                            f.writelines(strwrite+'\n')
                graph_out([],-1,'','')
                return p0, p0_corrmat
        else:
            if L_TKTS:
                print('Probabilities of successfully taking off with all tickets in hands: '+str(p0_tickets))
                if GRAPH:
                    title0 = 'Probabilities for flights setting off, with average positive cases: '+str(AVE)+', DAMP = '+str(DAMP)
                    graph_out(p0, 1, title0, color='blue')
                    graph_out([],-1,'','')
                if SAVE_FILE:
                    from time import asctime
                    with open(file = asctime().replace(' ','-').replace(':','')+'.txt', mode = 'a+', encoding = 'utf-8') as f:
                        f.writelines('SECTION| Probabilities of flights taking off, START HERE\n')
                        for idx_p0 in range(SPAN):
                            f.writelines('week '+str(idx_p0)+': '+str(p0[idx_p0])+'\n')
                        f.writelines('SECTION| Probabilities of successfully taking off with all tickets in hands\n')
                        f.writelines(str(p0_tickets)+'\n')
                return p0, p0_tickets
            else:
                if GRAPH:
                    title0 = 'Probabilities for flights setting off, with average positive cases: '+str(AVE)+', DAMP = '+str(DAMP)
                    graph_out(p0, 1, title0, color='blue')
                    graph_out([],-1,'','')
                if SAVE_FILE:
                    from time import asctime
                    with open(file = asctime().replace(' ','-').replace(':','')+'.txt', mode = 'a+', encoding = 'utf-8') as f:
                        f.writelines('SECTION| Probabilities of flights taking off, START HERE\n')
                        for idx_p0 in range(len(SPAN)):
                            f.writelines('week '+str(idx_p0)+': '+str(p0[idx_p0])+'\n')
                return p0

main()
