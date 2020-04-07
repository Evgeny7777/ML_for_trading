import math
import json

import pandas as pd

import py.mongo as mongo

def gth(df, key, value):
    return df[df[key] > value]

def move_column(df, col_name, place_no):
    if col_name not in df.columns: return df
    l = df.columns.to_list()
    l.insert(place_no, l.pop(l.index(col_name)))
    return df[l]

pd.DataFrame.gth = gth
pd.DataFrame.move_column = move_column

#========

def show_experiments():
    columns = ['id', 'name', 'status', 'description', 'created_at', 'updated_at', 'config_name', 'points']
    data = [(t.id, t.name, t.status, t.description, t.created_at, t.updated_at, t.configuration.name if t.configuration else None, len(t.points)) 
                for t in mongo.Experiment.objects]
    return pd.DataFrame(data, columns = columns).set_index('name')

def smartround(v, places=2):
    if math.isnan(v): return 0
    if v % 1 > 0: return round(v, places)
    return int(v)

def get_experiment_info(experiment_obj_id):
    ex = mongo.Experiment.objects(pk=experiment_obj_id).first()
    points = mongo.Point.objects(experiment=ex)

    all_points_dict_list = []
    for p in points:
        temp = json.loads(p.evaluation_on_val.to_json())
        _ = temp.pop('all_deals_test', None)
        _ = temp.pop('all_deals', None)
        val_stats = {'v'+k:v for k,v in temp.items()}
        # ===
        temp = json.loads(p.evaluation_on_test.to_json())
        _ = temp.pop('all_deals_test', None)
        _ = temp.pop('all_deals', None)
        test_stats = {'t'+k:v for k,v in temp.items()}
        #====
        p_dict = dict(step=p.step, **val_stats, **test_stats, **p.coordinates)
        p_dict = {k:smartround(v) for  k, v in p_dict.items()}
        p_dict['fine_tuned'] = p.fine_tuned
        p_dict['id'] = p.id
        all_points_dict_list.append(p_dict)

    #https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
    df = pd.DataFrame(all_points_dict_list)
    df = (df
        .assign(tdiff=df.tmean - df.tstd)
        .assign(vdiff=df.vmean - df.vstd)
        .set_index('step')
        .gth('tdeals_count', 20)
        .gth('vdeals_count', 20)
        .gth('vmean', 0)
        .sort_values('vprofit', ascending=False)
         )

    return df, points

def add_colors_and_move(df):
    return (df.move_column('tprofit', 0)
            .move_column('tmean', 1)
            .move_column('tstd', 2)
            .move_column('tdiff', 3)
            .move_column('tdeals_per_day', 4)
            .move_column('vprofit', 5)
            .move_column('vmean', 6)
            .move_column('vstd', 7)
            .move_column('vdiff', 8)
            .move_column('vdeals_per_day', 9)
            .move_column('fine_tuned', 1)
            .move_column('id', 1)
            .style.background_gradient('pink')
            .background_gradient('Greens', subset=['tprofit', 'tmean', 'tstd', 'tdiff', 'tdeals_per_day'])
            .background_gradient('Blues', subset=['vprofit', 'vmean', 'vstd', 'vdiff', 'vdeals_per_day'])
         )

def deals_from_step(points, step):
    p = None
    for pt in points:
        if pt.step == step: p = pt
    return pd.DataFrame(p.evaluation_on_test.all_deals)

def look_at_deals(points, interesting_steps):
    dfd = { v:deals_from_step(points, v) for v in interesting_steps}

    for k, v in dfd.items():
        print(f'Step {k}, sum {v.profit.sum()}, mean={v.profit.mean()}, std={v.profit.std()} deals={v.shape[0]}')
        v.profit.hist(bins=20, alpha=0.5, figsize=(50, 20))
        pass

    # overlap
    first_key=interesting_steps[0]
    df_over = dfd[first_key]
    for k, v in dfd.items():
        if k == first_key: continue
        df_over = df_over[df_over.date.isin(v.date)]

    print(f'Final: sum {df_over.profit.sum()}, mean={df_over.profit.mean()}, std={df_over.profit.std()} deals={df_over.shape[0]}')
    df_over.profit.hist(bins=20, alpha=0.5, figsize=(50, 20))