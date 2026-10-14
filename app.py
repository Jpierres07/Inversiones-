from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

st.set_page_config(page_title='Análisis de precios de acciones', layout='wide')
st.markdown("""<style>
.stApp {background:#0b1925;color:#e5f2f4;}
[data-testid="stSidebar"] {background:#102c3a;}
h1,h2,h3 {color:#d9f4f2;}
[data-testid="stMetric"] {background:#123342;border:1px solid #24505d;border-radius:12px;padding:16px;}
[data-testid="stMetricValue"] {color:#63ded0;}
[data-testid="stMetricLabel"] {color:#bfd7df;}
</style>""", unsafe_allow_html=True)
px.defaults.template = 'plotly_dark'
BASE = Path(__file__).resolve().parent
EMPRESAS = {'BVN': 'Buenaventura', 'SCCO': 'Southern Copper', 'BAP': 'Credicorp'}
METODOS = ['PMS', 'Ponderado', 'SES', 'Holt', 'Holt-Winters aditivo', 'Holt-Winters multiplicativo']
COLORES = dict(zip(METODOS, ['#f0ba68','#b4a1ec','#64cbe3','#63ded0','#ec8b8b','#8eaee3']))

@st.cache_data
def cargar(ruta):
    d = pd.read_csv(ruta, sep=';', encoding='utf-8-sig')
    d['Fecha'] = pd.to_datetime(d['Fecha'], errors='raise')
    if d['Fecha'].duplicated().any():
        raise ValueError(f'Fechas duplicadas en {Path(ruta).name}')
    d['Adj Close'] = pd.to_numeric(d['Adj Close'], errors='raise')
    if d['Adj Close'].isna().any() or not np.isfinite(d['Adj Close']).all() or (d['Adj Close'] <= 0).any():
        raise ValueError(f'Precios ausentes o no válidos en {Path(ruta).name}')
    return d.sort_values('Fecha').set_index('Fecha')


def ajustar(s, metodo, ventana):
    if metodo == 'PMS':
        return s.rolling(ventana).mean().shift(1), None
    if metodo == 'Ponderado':
        pesos = np.arange(1, ventana+1, dtype=float)
        return s.rolling(ventana).apply(lambda x: np.dot(x, pesos)/pesos.sum(), raw=True).shift(1), None
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter('always')
        if metodo == 'SES':
            modelo = SimpleExpSmoothing(s, initialization_method='estimated')
        elif metodo == 'Holt':
            modelo = Holt(s, initialization_method='estimated')
        else:
            modelo = ExponentialSmoothing(s, trend='add', seasonal='add' if metodo.endswith('aditivo') else 'mul', seasonal_periods=12, initialization_method='estimated')
        fit = modelo.fit(optimized=True)
    if any('converge' in str(a.message).lower() for a in avisos):
        raise ValueError('La optimización no convergió')
    if not np.isfinite(fit.fittedvalues).all():
        raise ValueError('Ajuste no finito')
    return pd.Series(np.asarray(fit.fittedvalues), index=s.index), fit


def errores(real, pred):
    e = np.asarray(real)-np.asarray(pred)
    return {'MAE': float(np.mean(abs(e))), 'RMSE': float(np.sqrt(np.mean(e**2))), 'MAPE (%)': float(np.mean(abs(e/np.asarray(real)))*100)}


def linea(series, titulo, eje='US$ por acción (cierre ajustado)'):
    f = go.Figure()
    for nombre, s in series.items():
        f.add_trace(go.Scatter(x=s.index, y=s.values, name=nombre, mode='lines'))
    f.update_layout(title=titulo, xaxis_title='Fecha', yaxis_title=eje, height=500, hovermode='x unified', dragmode='zoom', xaxis=dict(rangeslider=dict(visible=True), rangeselector=dict(buttons=[dict(count=1,label='1 año',step='year',stepmode='backward'),dict(count=5,label='5 años',step='year',stepmode='backward'),dict(label='Todo',step='all')])), template='plotly_dark', paper_bgcolor='#102633', plot_bgcolor='#102633')
    return f

try:
    mensuales = {t: cargar(BASE/'datos'/f'{t}_mensuales.csv')['Adj Close'] for t in EMPRESAS}
    diarios = {t: cargar(BASE/'datos'/f'{t}_diarios.csv')['Adj Close'] for t in EMPRESAS}
    datos = pd.concat(mensuales, axis=1).sort_index()
    inicio = max(s.index.min() for s in mensuales.values())
    fin = min(s.index.max() for s in mensuales.values())
    datos = datos.loc[inicio:fin].asfreq('MS')
    if datos.isna().any().any():
        raise ValueError('Hay meses ausentes en el periodo común; revisa los CSV.')
except Exception as exc:
    st.error(f'No se pudieron cargar los datos: {exc}')
    st.stop()

st.sidebar.title('Análisis de precios de acciones')
pagina = st.sidebar.radio('Contenido', ['Exploración de precios', 'Descomposición temporal', 'Estimación y errores', 'Vista de las tres empresas', 'Proyecciones futuras'])
ticker = st.sidebar.selectbox('Acción', list(EMPRESAS), format_func=lambda t: f'{t} | {EMPRESAS[t]}')
s = datos[ticker].copy()
st.sidebar.caption(f'{inicio:%m/%Y} a {fin:%m/%Y} | {len(datos)} registros mensuales por empresa')

if pagina == 'Exploración de precios':
    st.title('Comportamiento de las cotizaciones')
    frecuencia=st.radio('Frecuencia del gráfico',['Mensual','Diaria'],horizontal=True)
    base=datos if frecuencia=='Mensual' else pd.concat(diarios,axis=1)
    empresas=st.multiselect('Series visibles',list(EMPRESAS),default=[ticker])
    escala=st.radio('Forma de comparación',['Precio ajustado','Índice base 100'],horizontal=True)
    rango=st.date_input('Intervalo a explorar',value=(base.index.min().date(),base.index.max().date()),min_value=base.index.min().date(),max_value=base.index.max().date())
    x=base.loc[str(rango[0]):str(rango[1])] if len(rango)==2 else base
    if not empresas or x.empty:
        st.info('Selecciona al menos una empresa y un intervalo con datos.');st.stop()
    x=x[empresas]
    if escala=='Índice base 100':x=x.div(x.iloc[0]).mul(100)
    fig=linea({t:x[t] for t in empresas},'Trayectoria de las series','Índice base 100' if escala=='Índice base 100' else 'US$ por acción')
    if st.checkbox('Escala logarítmica',value=False):fig.update_yaxes(type='log')
    st.plotly_chart(fig,width='stretch',config={'displaylogo':False,'scrollZoom':True})
    st.caption('Arrastra para ampliar un tramo, usa el control inferior para desplazarte y pulsa una leyenda para ocultar una serie. Haz doble clic para recuperar la vista.')
    st.caption('En base 100, un valor de 120 representa un aumento del 20% frente al primer registro del intervalo. Las caídas o saltos aislados requieren revisión; el gráfico no identifica sus causas.')
    st.subheader('Distribución y valores extremos')
    vista=st.radio('Gráfico complementario',['Distribución de precios','Cambios mensuales'],horizontal=True)
    if vista=='Distribución de precios':
        st.plotly_chart(px.box(datos,points='outliers',labels={'variable':'Empresa','value':'Cierre ajustado (US$)'}),width='stretch')
        st.caption('Los puntos fuera de los límites del diagrama son candidatos a revisión. En una serie con tendencia, no implican necesariamente errores de registro.')
    else:
        cambios=datos.pct_change(fill_method=None).mul(100).dropna()
        st.plotly_chart(linea({t:cambios[t] for t in empresas},'Variaciones entre meses consecutivos','Cambio (%)'),width='stretch')
    st.dataframe(datos[empresas].describe().T, width='stretch')

elif pagina == 'Descomposición temporal':
    st.title(f'Componentes de la serie de {ticker}')
    tipo=st.radio('Estructura',['Aditiva','Multiplicativa'],horizontal=True)
    des=seasonal_decompose(s,model='additive' if tipo=='Aditiva' else 'multiplicative',period=12)
    componentes=pd.DataFrame({'Observado':s,'Tendencia':des.trend,'Estacionalidad':des.seasonal,'Residuo':des.resid})
    from plotly.subplots import make_subplots
    fig=make_subplots(rows=4,cols=1,shared_xaxes=True,subplot_titles=list(componentes),vertical_spacing=.07)
    for i,c in enumerate(componentes,1):
        fig.add_trace(go.Scatter(x=s.index,y=componentes[c],name=c,mode='lines'),row=i,col=1)
    fig.update_layout(height=850,template='plotly_dark',showlegend=False,hovermode='x unified')
    st.plotly_chart(fig,width='stretch',config={'displaylogo':False,'scrollZoom':True})
    st.latex(r'Y_t=T_t+E_t+R_t' if tipo=='Aditiva' else r'Y_t=T_t\times E_t\times R_t')
    st.caption('Se usa un ciclo de 12 meses. La tendencia resume cambios prolongados; la estacionalidad estima un patrón anual; el residuo recoge lo restante. Los extremos de la tendencia quedan sin estimar por el promedio centrado.')
    st.info('En la estructura aditiva, los componentes se expresan en dólares por acción. En la multiplicativa, estacionalidad y residuo son factores: 1 indica ausencia de efecto. La descomposición no demuestra por sí sola una estacionalidad estable.')
    st.download_button('Exportar componentes',componentes.to_csv().encode('utf-8-sig'),f'{ticker}_descomposicion.csv')

elif pagina == 'Estimación y errores':
    st.title(f'Ajuste de métodos para {ticker}')
    ventana=st.slider('Ventana de promedios móviles (meses)',3,24,12)
    st.caption('Se utilizan todos los meses como entrenamiento. Los errores se comparan en los mismos registros, excluyendo los primeros 24 meses para reducir el efecto de inicialización.')
    ajustes={};filas=[];fallos={};comienzo=max(24,ventana)
    with st.spinner('Estimando métodos...'):
        for m in METODOS:
            try:
                f,fit=ajustar(s,m,ventana)
                ajustes[m]=f
                filas.append({'Método':m,**errores(s.iloc[comienzo:],f.iloc[comienzo:]),'Meses evaluados':len(s)-comienzo})
            except Exception as exc:fallos[m]=str(exc)
    if not filas:st.error('No se pudieron calcular los métodos.');st.stop()
    tabla=pd.DataFrame(filas).sort_values('RMSE')
    visibles=st.multiselect('Métodos visibles',list(ajustes),default=[tabla.iloc[0]['Método']])
    st.plotly_chart(linea({'Observado':s,**{m:ajustes[m] for m in visibles}},'Valores observados y ajustes'),width='stretch',config={'displaylogo':False,'scrollZoom':True})
    st.dataframe(tabla,hide_index=True,width='stretch')
    ganador=tabla.iloc[0]
    st.info(f'{ganador["Método"]} presenta el menor RMSE de entrenamiento: US$ {ganador["RMSE"]:,.2f} por acción. Este resultado describe el ajuste al historial; no evalúa precisión futura.')
    st.caption('MAE: error absoluto medio. RMSE: penaliza más los errores grandes. Ambos están en dólares por acción. MAPE: error absoluto porcentual medio. Menores valores indican mejor ajuste en los registros evaluados.')
    with st.expander('Cómo se calculan los métodos'):
        st.write('El promedio móvil simple asigna el mismo peso a los meses anteriores. El ponderado asigna pesos de 1 a la ventana, dando más importancia al mes más reciente. Ambos se desplazan un mes para evitar usar el precio que se intenta estimar. SES estima nivel; Holt incorpora tendencia; Holt-Winters incorpora tendencia y estacionalidad de 12 meses, aditiva o multiplicativa. Los parámetros exponenciales se optimizan por suma de errores cuadrados.')
    if fallos:st.warning('Métodos sin estimación: '+str(fallos))
    st.download_button('Exportar errores de entrenamiento',tabla.to_csv(index=False).encode('utf-8-sig'),f'{ticker}_errores_entrenamiento.csv')

elif pagina == 'Vista de las tres empresas':
    st.title('BVN, SCCO y BAP en una misma vista')
    analisis=st.selectbox('Análisis conjunto',['Serie histórica','Descomposición aditiva','Descomposición multiplicativa','Promedios móviles','Suavizamiento exponencial','Errores de entrenamiento'])
    ventana=st.slider('Ventana común de promedios móviles (meses)',3,24,12)
    if analisis=='Serie histórica':
        escala=st.radio('Escala',['Precio ajustado','Base 100'],horizontal=True)
        x=datos if escala=='Precio ajustado' else datos.div(datos.iloc[0]).mul(100)
        st.plotly_chart(linea({t:x[t] for t in EMPRESAS},'Evolución de las tres empresas','US$ por acción' if escala=='Precio ajustado' else 'Índice base 100'),width='stretch')
        st.caption('La base 100 facilita comparar variaciones porcentuales desde el mismo mes inicial.')
    elif analisis.startswith('Descomposición'):
        componente=st.selectbox('Componente a comparar',['Observado','Tendencia','Estacionalidad','Residuo'])
        for col,t in zip(st.columns(3),EMPRESAS):
            des=seasonal_decompose(datos[t],model='additive' if analisis.endswith('aditiva') else 'multiplicative',period=12)
            componentes={'Observado':des.observed,'Tendencia':des.trend,'Estacionalidad':des.seasonal,'Residuo':des.resid}
            with col:
                st.subheader(t)
                st.plotly_chart(linea({componente:componentes[componente]},componente,'Factor' if analisis.endswith('multiplicativa') and componente in ['Estacionalidad','Residuo'] else 'US$ por acción'),width='stretch')
    else:
        tablas=[]
        for col,t in zip(st.columns(3),EMPRESAS):
            x=datos[t];ajustes={};filas=[]
            for m in METODOS:
                try:
                    f,_=ajustar(x,m,ventana);ajustes[m]=f
                    filas.append({'Empresa':t,'Método':m,**errores(x.iloc[24:],f.iloc[24:])})
                except Exception as exc:
                    with col:st.caption(f'{m}: {exc}')
            tabla=pd.DataFrame(filas);tablas.append(tabla)
            with col:
                st.subheader(t)
                if analisis=='Errores de entrenamiento':
                    st.dataframe(tabla.drop(columns='Empresa').sort_values('RMSE'),hide_index=True,width='stretch')
                else:
                    elegidos=['PMS','Ponderado'] if analisis=='Promedios móviles' else METODOS[2:]
                    st.plotly_chart(linea({'Observado':x,**{m:ajustes[m] for m in elegidos if m in ajustes}},analisis),width='stretch')
        total=pd.concat(tablas,ignore_index=True)
        st.caption('Todos los errores se calculan sobre los mismos meses: desde el registro 25 hasta el último. RMSE y MAE se comparan entre métodos de una misma empresa; para contrastar empresas con precios distintos, observa también MAPE.')
        st.download_button('Exportar resultados de las tres empresas',total.to_csv(index=False).encode('utf-8-sig'),'errores_tres_empresas.csv')

elif pagina == 'Proyecciones futuras':
    st.title('Proyecciones mensuales de precios de acciones')
    st.caption('Pronósticos exploratorios. No constituyen recomendaciones de inversión.')

    empresa = st.selectbox('Empresa', list(EMPRESAS),
                           index=list(EMPRESAS).index(ticker),
                           format_func=lambda t: f'{EMPRESAS[t]} ({t})')
    metodo = st.selectbox('Modelo de pronóstico',
                          ['Holt-Winters multiplicativo',
                           'Holt-Winters aditivo', 'Holt', 'SES'])
    horizonte = st.selectbox('Meses a proyectar', [3, 6, 12], index=1)
    meses_historicos = st.slider('Meses históricos visibles', 12, 120, 36, 12)

    serie = datos[empresa].dropna().astype(float).asfreq('MS')
    if serie.isna().any() or (serie <= 0).any():
        st.error('La serie contiene meses faltantes o valores no positivos.')
        st.stop()

    @st.cache_data(show_spinner=False)
    def calcular_proyeccion(valores, fechas, nombre_metodo, pasos):
        s = pd.Series(valores, index=pd.to_datetime(fechas), dtype=float).asfreq('MS')
        if nombre_metodo == 'SES':
            modelo = SimpleExpSmoothing(s, initialization_method='estimated')
        elif nombre_metodo == 'Holt':
            modelo = Holt(s, initialization_method='estimated')
        elif nombre_metodo == 'Holt-Winters aditivo':
            modelo = ExponentialSmoothing(
                s, trend='add', seasonal='add', seasonal_periods=12,
                initialization_method='estimated')
        else:
            modelo = ExponentialSmoothing(
                s, trend='add', seasonal='mul', seasonal_periods=12,
                initialization_method='estimated')
        ajuste = modelo.fit(optimized=True)
        return ajuste.forecast(pasos)

    try:
        futuro = calcular_proyeccion(tuple(serie.to_numpy()),
                                    tuple(serie.index), metodo, horizonte)
    except Exception as exc:
        st.error(f'No se pudo ajustar el modelo: {exc}')
        st.stop()

    fechas_futuras = pd.date_range(serie.index[-1] + pd.offsets.MonthBegin(1),
                                   periods=horizonte, freq='MS')
    futuro = pd.Series(np.asarray(futuro, dtype=float),
                      index=fechas_futuras, name='Precio proyectado (USD)')
    if not np.isfinite(futuro.to_numpy()).all() or (futuro <= 0).any():
        st.warning('El modelo generó precios no válidos. Prueba otro método.')
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric('Último cierre ajustado', f'US$ {serie.iloc[-1]:,.2f}')
    c2.metric('Proyección final', f'US$ {futuro.iloc[-1]:,.2f}')
    cambio = (futuro.iloc[-1] / serie.iloc[-1] - 1) * 100
    c3.metric('Variación proyectada', f'{cambio:+.2f}%')

    grafico = go.Figure()
    pasado = serie.tail(meses_historicos)
    grafico.add_trace(go.Scatter(x=pasado.index, y=pasado.values,
                                 name='Histórico', mode='lines'))
    # Conectar la proyección con el último precio real
    x_proy = [serie.index[-1], *futuro.index]
    y_proy = [float(serie.iloc[-1]), *futuro.to_list()]
    grafico.add_trace(go.Scatter(x=x_proy, y=y_proy,
                                 name=f'Pronóstico: {metodo}', mode='lines+markers',
                                 line=dict(dash='dash')))
    grafico.add_shape(type='line', x0=serie.index[-1], x1=serie.index[-1],
                      y0=0, y1=1, yref='paper', line=dict(color='gray', dash='dot'))
    grafico.update_layout(title=f'{EMPRESAS[empresa]} ({empresa}): histórico y pronóstico',
                          xaxis_title='Mes', yaxis_title='US$ por acción (Adj Close)',
                          template='plotly_dark', hovermode='x unified', height=500)
    st.plotly_chart(grafico, width='stretch')

    tabla = pd.DataFrame({'Mes': futuro.index.strftime('%Y-%m'),
                          'Precio proyectado (USD)': futuro.to_numpy(),
                          'Variación frente al último cierre (%)':
                              (futuro.to_numpy() / serie.iloc[-1] - 1) * 100})
    st.dataframe(tabla.style.format({'Precio proyectado (USD)': '{:,.2f}',
                                    'Variación frente al último cierre (%)': '{:+.2f}%'}),
                 width='stretch', hide_index=True)
    st.download_button('Descargar proyecciones CSV',
                       data=tabla.to_csv(index=False).encode('utf-8-sig'),
                       file_name=f'pronostico_{empresa}_{metodo.replace(" ", "_")}_{horizonte}m.csv',
                       mime='text/csv')
    st.info('Estas proyecciones son extrapolaciones estadísticas, no precios garantizados. '
            'El RMSE del informe es de entrenamiento; evalúa pronósticos fuera de muestra '
            'antes de elegir el modelo definitivo.')


st.divider()
st.caption('Fuente: Yahoo Finance | Variable: cierre ajustado | Datos históricos hasta septiembre de 2026')