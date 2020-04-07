from IPython.core.display import display, HTML, Markdown
import pandas as pd

def strip_all_strings(df):
    df_obj = df.select_dtypes(include=['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

def h(s, level, color=None):
    if color:
        line = f'<font color="{color}">{s}</font>'
    else:
        line = s
    
    replacements = {
        '<red>': '<font color="red">',
        '<blue>': '<font color="blue">',
        '<green>': '<font color="green">',
        '<orange>': '<font color="orange">',
        '</>': '</font>'
    }
    for r in replacements:
        line = line.replace(r, replacements[r])
    
    line = '#'*level + ' ' + line
    display(Markdown(line))
    
def h1(s, color=None):
    h(s, 1, color)

def h2(s, color=None):
    h(s, 2, color)

def h3(s, color=None):
    h(s, 3, color)

def h4(s, color=None):
    h(s, 4, color)

def print_ds(ds, names=['','']):
    ds = ds.reset_index()
    ds.columns = names
    display(HTML(ds.to_html(index=False)))