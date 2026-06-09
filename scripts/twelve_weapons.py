#!/usr/bin/env python3
"""
Twelve Weapons — 统计反算检测器库
====================================
Heaven's Net 统计反算武器集合。从 Mean±SD 和声称的 P 值出发，
反算并验证论文中报告的统计结果是否真实可达。

Supported weapons:
    3.  CV (Coefficient of Variation) anomaly detection
    5.  Independent samples t-test recalculation
    6.  One-way ANOVA recalculation from summary data
    7.  Chi-square / Fisher's exact test recalculation
    11. P-value distribution analysis

Usage:
    from twelve_weapons import TwelveWeapons, analyze_paper
    
    result = analyze_paper(
        mean_sd_pairs=[(1.2, 0.3), (1.8, 0.4)],
        sample_sizes=[6, 6],
        p_values=[0.03, 0.12, 0.045],
        data_type='general'
    )
    
    # Or use the class wrapper:
    weapons = TwelveWeapons()
    result = weapons.analyze_paper(...)

Dependencies: math (stdlib only — zero external dependencies)
"""

import os, re, math, json
from collections import Counter
from math import erf, sqrt

def norm_cdf(x):
    return 0.5 * (1 + erf(x / 1.4142135623730951))

def p_from_t(t, df):
    x = t * (1 - 1/(4*df)) / sqrt(1 + t*t/(2*df))
    return 2 * (1 - norm_cdf(abs(x)))

def p_from_f(f, df1, df2):
    if f <= 0: return 1.0
    z = (f**(1/3)*(1-2/(9*df2)) - (1-2/(9*df1))) / sqrt(2/(9*df1)*f**(2/3) + 2/(9*df2))
    return 2 * (1 - norm_cdf(abs(z)))

CV_THRESHOLDS = {
    'mtt_od': {'normal': (5, 20), 'impossible': 1.0, 'desc': 'MTT/CCK8 OD值'},
    'cell_count': {'normal': (5, 15), 'impossible': 1.0, 'desc': '细胞计数/增殖'},
    'wb_gray': {'normal': (10, 30), 'impossible': 2.0, 'desc': 'Western Blot灰度'},
    'flow_apoptosis': {'normal': (5, 20), 'impossible': 1.5, 'desc': '流式凋亡率'},
    'clinical_age': {'normal': (15, 40), 'impossible': 1.0, 'desc': '年龄'},
    'clinical_lab': {'normal': (5, 30), 'impossible': 1.0, 'desc': '血常规/生化'},
    'migration': {'normal': (5, 25), 'impossible': 1.0, 'desc': '细胞迁移/侵袭'},
    'general': {'normal': (5, 30), 'impossible': 1.0, 'desc': '通用生物测量'},
}

def weapon_cv(means, sds, data_type='general'):
    """武器3: CV检测"""
    results = []
    threshold = CV_THRESHOLDS.get(data_type, CV_THRESHOLDS['general'])
    for mv, sv in zip(means, sds):
        if mv <= 0 or sv < 0: continue
        cv = abs(sv/mv*100)
        if cv == 0: results.append({'level': 'CRITICAL', 'reason': f'SD=0: mean={mv}'})
        elif cv < threshold['impossible']: results.append({'level': 'CRITICAL', 'reason': f'CV={cv:.2f}% < {threshold["impossible"]}%'})
        elif cv < threshold['normal'][0]*0.5: results.append({'level': 'SUSPICIOUS', 'reason': f'CV={cv:.2f}%偏低'})
    return results

def weapon_t_test(mean1, sd1, n1, mean2, sd2, n2):
    """武器5: t检验完全反算"""
    if sd1 <= 0 or sd2 <= 0: return {'error': 'SD=0'}
    sp = sqrt(((n1-1)*sd1**2 + (n2-1)*sd2**2) / (n1+n2-2))
    se = sp * sqrt(1/n1 + 1/n2)
    t = (mean1-mean2)/se if se > 1e-10 else float('inf')
    df = n1+n2-2
    p = p_from_t(t, df)
    d = abs(mean1-mean2)/sp if sp>0 else 0
    effect = '微小' if d<0.2 else '小' if d<0.5 else '中等' if d<0.8 else '大' if d<2.0 else '极大(可疑)'
    return {'t':t, 'df':df, 'p':p, 'cohens_d':d, 'effect':effect, 'extreme': d>5.0}

def weapon_anova(groups):
    """武器6: ANOVA反算。groups=[(mean,sd,n), ...]"""
    k = len(groups)
    N = sum(g[2] for g in groups)
    gm = sum(g[0]*g[2] for g in groups)/N
    SSb = sum(g[2]*(g[0]-gm)**2 for g in groups)
    SSw = sum((g[2]-1)*g[1]**2 for g in groups)
    F = (SSb/(k-1))/(SSw/(N-k)) if SSw>0 and k>1 else 0
    p = p_from_f(F, k-1, N-k)
    eta_sq = SSb/(SSb+SSw)
    n_too_small = any(g[2]<=3 for g in groups)
    return {'F':F, 'p':p, 'eta_sq':eta_sq, 'n_too_small':n_too_small}

def weapon_chisq(x1, n1, x2, n2):
    """武器7: 卡方反算。x可以是例数或率(0-1)"""
    if x1<1 and x1>0: x1 = round(x1*n1)
    if x2<1 and x2>0: x2 = round(x2*n2)
    a,b = x1, n1-x1
    c,d = x2, n2-x2
    N = n1+n2
    E = [n1*(a+c)/N, n1*(b+d)/N, n2*(a+c)/N, n2*(b+d)/N]
    need_fisher = min(E) < 5
    if not need_fisher:
        chisq = N*(abs(a*d-b*c)-N/2)**2/(n1*n2*(a+c)*(b+d))
        p = p_from_chisq(chisq, 1)
    else: chisq, p = None, None
    return {'chisq':chisq, 'p':p, 'need_fisher':need_fisher, 'min_expected':min(E)}

def p_from_chisq(chisq, df):
    if df<=0 or chisq<=0: return 1.0
    z = ((chisq/df)**(1/3)-(1-2/(9*df)))/sqrt(2/(9*df))
    return 1-norm_cdf(z)

def weapon_p_distribution(p_values, method='declared'):
    """武器11: P值分布分析"""
    p_num = [float(p) for p in p_values if 0<float(p)<0.2]
    if len(p_num) < 5: return {'signal': False}
    p_005 = sum(1 for p in p_num if 0.04<p<0.06)
    single_threshold = Counter([round(float(p),2) for p in p_values])
    dominant = single_threshold.most_common(1)[0]
    return {
        'p_near_005': p_005, 'total': len(p_num),
        'ratio_005': p_005/len(p_num),
        'single_dominant': dominant[1]/len(p_num) > 0.5,
        'dominant_value': dominant[0]
    }

def analyze_paper(mean_sd_pairs, sample_sizes, p_values, data_type='general', methods_declared='', n_groups_desc=''):
    """一站式分析：给定论文数据，返回所有武器检测结果"""
    means = [float(m) for m,s in mean_sd_pairs]
    sds = [float(s) for m,s in mean_sd_pairs]

    results = {}
    cv_findings = weapon_cv(means, sds, data_type)
    results['cv'] = {'critical': sum(1 for c in cv_findings if c['level']=='CRITICAL')}
    results['sd_zero'] = sum(1 for s in sds if s < 0.001)

    if len(means)>=2 and sample_sizes:
        n = sample_sizes[0]
        t_result = weapon_t_test(means[0], sds[0], n, means[1], sds[1], n)
        results['t_test'] = t_result

    p_analysis = weapon_p_distribution(p_values)
    results['p_dist'] = p_analysis

    score = results['cv']['critical']*3 + results['sd_zero']*3
    if p_analysis.get('single_dominant'): score += 2
    if results.get('t_test',{}).get('extreme'): score += 2

    results['score'] = score
    results['verdict'] = 'FAKE' if score>=3 else 'SUSPECT' if score>=1 else 'CLEAN'
    return results


class TwelveWeapons:
    """统计反算12武器兼容包装"""
    
    def __init__(self):
        pass
    
    def analyze_paper(self, mean_sd_pairs, sample_sizes, p_values, data_type='general', methods_declared='', n_groups_desc=''):
        return analyze_paper(mean_sd_pairs, sample_sizes, p_values, data_type, methods_declared, n_groups_desc)


if __name__ == '__main__':
    print("12武器统计学反算库已就绪。")
    print("使用: analyze_paper(mean_sd_pairs, sample_sizes, p_values, data_type='general')")
