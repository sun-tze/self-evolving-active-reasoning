"""Render all frozen primary comparisons from the sanitized release."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
    r=json.loads((a.directory/'results.json').read_text(encoding='utf8'))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(13.2,5.5),gridspec_kw={'width_ratios':[1,1.15]})
    fig.patch.set_facecolor('#f8f7f3')
    colors={'B0':'#68777c','R0':'#b9bdb5','R1':'#b9bdb5','E0':'#187b78','E1':'#187b78'}
    labels={'B0':'No note','R0':'Fitness-only R0','R1':'Fitness-only R1','E0':'Trace-guided E0','E1':'Trace-guided E1'}
    order=['B0','R0','R1','E0','E1'];s=r['arm_summaries']['pooled']
    ax=axes[0]
    ax.barh(range(5),[s[k]['mean_capped25'] for k in order],color=[colors[k] for k in order],height=.65)
    ax.set_yticks(range(5),[labels[k] for k in order]);ax.invert_yaxis();ax.set_xlim(0,14.2)
    for i,k in enumerate(order):
        ax.text(s[k]['mean_capped25']+.13,i,f"{s[k]['mean_capped25']:.2f}  ({s[k]['resolved']}/200)",va='center',fontsize=10)
    ax.set_xlabel('Mean queries, capped at 25 (lower is better)')
    ax.set_title('All five frozen policies · 200 new cases',loc='left',fontweight='bold',fontsize=12,pad=15)
    ax.text(0,-.17,'Parentheses: identified by the public monitor.',transform=ax.transAxes,fontsize=9,color='#45545a')
    ax=axes[1];names=[]
    for i,(c,k) in enumerate([('0','Emean-B0'),('0','Emean-Rmean'),('1','Emean-B0'),('1','Emean-Rmean')]):
        v=r['statistics']['cohort_results'][c]['contrasts'][k];x=v['mean_difference'];lo,hi=v['CI98_75_primary']
        ax.errorbar(x,i,xerr=[[x-lo],[hi-x]],fmt='o',color='#187b78',capsize=5,lw=2,markersize=6)
        names.append(f"Cohort {int(c)+1}: E − {'no note' if k=='Emean-B0' else 'fitness-only'}")
        ax.annotate(f'{x:.2f} [{lo:.2f}, {hi:.2f}]',(x,i),xytext=(0,-20),textcoords='offset points',ha='center',fontsize=9)
    ax.set_yticks(range(4),names);ax.invert_yaxis();ax.set_ylim(3.6,-.6);ax.axvline(0,color='#9c4741',ls='--',lw=1)
    ax.set_xlim(-6, .4);ax.set_xlabel('Paired query difference · negative favors E')
    ax.set_title('Four predeclared primary comparisons',loc='left',fontweight='bold',fontsize=12,pad=15)
    for ax in axes:
        ax.set_facecolor('#f8f7f3');ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.13);ax.set_axisbelow(True)
    fig.suptitle('Trace-guided proposal + performance selection reduced query cost',x=.05,ha='left',fontsize=16,fontweight='bold',color='#17333b')
    fig.text(.05,.02,'98.75% primary intervals: approximate four-comparison family 95%. Two learning blocks; two cohorts of the same frozen policies.\nPublic-monitor identification, not model final-answer accuracy. Earlier negative study retained. No broad optimizer-reliability claim.',fontsize=9,color='#45545a')
    fig.subplots_adjust(left=.15,right=.98,top=.82,bottom=.24,wspace=.8)
    for ext in ['png','svg']:
        fig.savefig(a.directory/f'results-figure.{ext}',dpi=180,facecolor=fig.get_facecolor())
    plt.close(fig)

if __name__=='__main__':main()
