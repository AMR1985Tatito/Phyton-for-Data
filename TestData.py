import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, MetaData, Table, select, func, outerjoin
from sqlalchemy.orm import sessionmaker
from sqlalchemy import extract, cast, Date
from datetime import timedelta

# Conexion BBDD
db_server = 'localhost'
db_user = 'postgres'
db_port = '5432'
db_password = 'Vempakus_7870'
db_name = 'Northwind'

conexion_string = f'postgresql://{db_user}:{db_password}@{db_server}:{db_port}/{db_name}'
engine = create_engine(conexion_string)
bd_datos = MetaData()

Session = sessionmaker(bind=engine)
session = Session()

#------------------------------Ejercicio 1. Familiarizarse con la Base de Datos------------------------------

# Obtener Info BBDD
bd_datos.reflect(bind=engine)
table_names = sorted(bd_datos.tables.keys())
print(f"---------- Las tablas de la BBDD son las siguientes ----------")
for table_name in table_names:
    print(table_name)
print()

# Obtener las Relaciones entre Tablas
print(f"---------- Las Relaciones entre Tablas de de la BBDD son las siguientes ----------")
for table_name in bd_datos.tables:
    table = bd_datos.tables[table_name]
    for fk in table.foreign_keys:
        
        print(f"Tabla '{table_name}' tiene una relación con '{fk.target_fullname}'. Claves: {fk.constraint.columns.keys()}")
print()
#------------------------------Ejercicio 2. Primeras consultas------------------------------

# Defininicion Tablas
employees = Table('employees', bd_datos, autoload=True,)
products = Table('products', bd_datos, autoload=True, autoload_with=engine)
suppliers = Table('suppliers', bd_datos, autoload=True)
orders = Table('orders', bd_datos, autoload=True)
order_details = Table('order_details', bd_datos, autoload=True)
customers = Table('customers', bd_datos, autoload=True)
shippers = Table('shippers', bd_datos, autoload=True)
categories = Table('categories', bd_datos,autoload=True)

# Consulta 1: Empleados contratados en 'Global Importaciones'
print(f"---------- EJERCICIO 2 - Consulta 1: Empleados Contratados en 'Global Importaciones ----------")

numero_empleados = select(func.count()).select_from(employees)
resultados_numero_empleados = session.execute(numero_empleados).scalar()
print(f"Número total de empleados: {resultados_numero_empleados}")
print()

query_empleados = select(employees)
resultados_empleados = session.execute(query_empleados)
for empleado in resultados_empleados:
    print(f"ID: {empleado[0]} - Empleado: {empleado[2]} {empleado[1]} - Ciudad: {empleado[8]} - Pais: {empleado[11]}")
print()

# Consulta 2: Listado de Productos'
print(f"---------- EJERCICIO 2 - Consulta 2: Listado de Productos ----------")
query_productos = select(products)
resultados_productos = session.execute(query_productos)
for product in resultados_productos:
    print(f"ID Producto: {product[0]} - ID Proveedor: {product[2]} - Nombre Producto: {product[1]} - Precio Unidad: {product[5]} - Unidades Stock: {product[6]} - Unidades Pedidas: {product[7]} - Producto Discontinuado: {product[9]}")
    print()
print()

# Consulta 3: Productos Discontinuados'
print(f"---------- EJERCICIO 2 - Consulta 3: Productos Discontinuados ----------")
numero_discontinuados = select(func.count()).where(products.c.discontinued == 1)
resultados_numero_discontinuados = session.execute(numero_discontinuados).scalar()

print(f"Número total de Productos Discontinuados es: {resultados_numero_discontinuados}")
print()

productos_descontinuados = select(products).where(products.c.discontinued == 1)
resultados_productos_descontinuados = session.execute(productos_descontinuados)
for producto in resultados_productos_descontinuados:
       print(f"Producto: {producto[1]} - Cantidad en Stock: {producto[6]}")
print()

# Consulta 4: Proveedores'
print(f"---------- EJERCICIO 2 - Consulta 4: Proveedores ----------")
numero_proveedores = select(func.count()).select_from(suppliers)
resultados_numero_proveedores = session.execute(numero_proveedores).scalar()

print(f"Número total de Proveedores es: {resultados_numero_proveedores}")
print()

proveedores = select(suppliers)
resultados_proveedores = session.execute(proveedores)
for proveedor in resultados_proveedores:
       print(f"ID: {proveedor[0]} - Empresa: {proveedor[1]} - Ciudad: {proveedor[5]} - Pais: {proveedor[8]}")
print()

# Consulta 5: Pedidos'
print(f"---------- EJERCICIO 2 - Consulta 5: Pedidos ----------")
numero_pedidos = select(func.count()).select_from(orders)
resultados_numero_pedidos = session.execute(numero_pedidos).scalar()

print(f"Número total de Pedidos es: {resultados_numero_pedidos}")
print()

pedidos = select(orders)
resultados_pedidos = session.execute(pedidos)
for pedido in resultados_pedidos:
       print(f"ID Pedido: {pedido[0]} - ID Cliente: {pedido[2]} - ID Trasportista: {pedido[1]} - Dia Pedido: {pedido[3]} - Dia Requerido Llegada: {pedido[4]} - Dia LLegada: {pedido[5]}")
       print()
print()

# Consulta 6: Clientes'
print(f"---------- EJERCICIO 2 - Consulta 6: Clientes ----------")
numero_clientes = select(func.count()).select_from(customers)
resultados_numero_clientes = session.execute(numero_clientes).scalar()

print(f"Número total de Clientes es: {resultados_numero_clientes}")
print()

clientes = select(customers)
resultados_clientes = session.execute(clientes)
for cliente in resultados_clientes:
       print(f"ID Cliente: {cliente[0]} - Nombre: {cliente[1]} - Ciudad: {cliente[5]} - Pais: {cliente[8]}")
print()

# Consulta 7: Trasporte'
print(f"---------- EJERCICIO 2 - Consulta 7: Empresas Trasporte ----------")
numero_trasporte = select(func.count()).select_from(shippers)
resultados_numero_trasporte = session.execute(numero_trasporte).scalar()

print(f"Número total de Empresas de Trasporte es: {resultados_numero_trasporte}")
print()

trasporte = select(shippers)
resultados_trasporte = session.execute(trasporte)
for trasporte in resultados_trasporte:
       print(f"ID: {trasporte[0]} - Nombre: {trasporte[1]}")
print()

#------------------------------Ejercicio 3. Análisis de la Empresa------------------------------
print(f"---------- EJERCICIO 3 - Analisis de la Empresa ----------")
print()

# Crear DataFrame de Pedidos y Clientes
print(f"---------- EJERCICIO 3 - Consulta 1: Evolucion Pedidos ----------")

consulta_pedidos_clientes = select(
    orders.c.order_id,
    orders.c.customer_id,
    orders.c.ship_via,
    orders.c.order_date,
    orders.c.required_date,
    orders.c.shipped_date,
    customers.c.company_name.label('customer_company'),
    customers.c.contact_name.label('customer_contact'),
    customers.c.city.label('customer_city'),
    customers.c.country.label('customer_country')
).where(orders.c.customer_id == customers.c.customer_id)

df_pedidos_clientes = pd.read_sql(consulta_pedidos_clientes, engine)

print("DataFrame de Pedidos y Clientes:")
print(df_pedidos_clientes.head())
print()

# Crear DataFrame de Productos, Proveedores y Detalles de Pedidos
consulta_productos_proveedores_pedidos = select(
    products.c.product_id,
    products.c.product_name,
    products.c.unit_price,
    products.c.units_in_stock,
    suppliers.c.company_name.label('supplier_company'),
    suppliers.c.contact_name.label('supplier_contact'),
    suppliers.c.city.label('supplier_city'),
    suppliers.c.country.label('supplier_country'),
    order_details.c.order_id,
    order_details.c.quantity,
    order_details.c.unit_price.label('order_unit_price'),
    order_details.c.discount
).where(products.c.supplier_id == suppliers.c.supplier_id).where(products.c.product_id == order_details.c.product_id)

df_productos_proveedores_pedidos = pd.read_sql(consulta_productos_proveedores_pedidos, engine)

print("DataFrame de Productos, Proveedores y Detalles de Pedidos:")
print(df_productos_proveedores_pedidos.head())
print()

# Obtener el número de pedidos por mes y año
consulta_pedidos_mes = select(
    extract('year', orders.c.order_date).label('anio'),
    extract('month', orders.c.order_date).label('mes'),
    func.count().label('num_pedidos')
).group_by('anio', 'mes').order_by('anio', 'mes')

resultados_pedidos_mes = session.execute(consulta_pedidos_mes).fetchall()

print("Número de Pedidos por Mes y Año:")
for resultado in resultados_pedidos_mes:
    print(f"Año: {resultado.anio}, Mes: {resultado.mes}, Pedidos: {resultado.num_pedidos}")
print()

# Lista Temporal
anios = []
meses = []
num_pedidos = []

for resultado in resultados_pedidos_mes:
    anios.append(resultado.anio)
    meses.append(resultado.mes)
    num_pedidos.append(resultado.num_pedidos)

# Crear la figura y los ejes
fig, ax = plt.figure(figsize=(12, 6), facecolor='#ffffff'), plt.gca()
ax.set_facecolor('none')

plt.plot(range(len(num_pedidos)), num_pedidos, marker='o', linestyle='-', color='#219ebc')

# Personalizar el gráfico
plt.title('Evolución de Pedidos por Mes y Año', fontsize=16, color='#fb8500')
plt.xlabel('Meses', fontsize=12, color='#fb8500')
plt.ylabel('Número de Pedidos', fontsize=12, color='#fb8500')
plt.xticks(range(len(num_pedidos)), [f"{mes}/{anio}" for mes, anio in zip(meses, anios)], rotation=45, fontsize=10, color='#219ebc')
plt.yticks(fontsize=10, color='#219ebc')
plt.grid(False)

# Mostrar la gráfica
plt.tight_layout()
plt.show()

# Paises con mas Ventas
print(f"---------- EJERCICIO 3 - Consulta 2: Paises con mas Ventas ----------")

continentes = {
    'Europe': ['Austria', 'Belgium', 'Denmark', 'Finland', 'France', 'Germany', 'Ireland', 'Italy', 'Norway', 'Poland', 'Portugal', 'Spain', 'Sweden', 'Switzerland', 'UK'],
    'America': ['Argentina', 'Brazil', 'Canada', 'Mexico', 'USA', 'Venezuela']
}

def obtener_continente(pais):
    for continente, paises in continentes.items():
        if pais in paises:
            return continente
    return 'Otro'

df_pedidos_clientes['continente'] = df_pedidos_clientes['customer_country'].apply(obtener_continente)
pedidos_por_continente = df_pedidos_clientes['continente'].value_counts()

# Crear la figura y los ejes
fig, ax = plt.subplots(figsize=(10, 6), facecolor='#ffffff')
ax.set_facecolor('none')
pedidos_por_continente.plot(kind='bar', ax=ax, color=['#219ebc', '#023047', '#8ecae6'])

# Personalizar el gráfico
plt.title('Distribución de Pedidos por Continente', fontsize=16, color='#fb8500')
plt.xlabel('Continente', fontsize=12, color='#fb8500')
plt.ylabel('Número de Pedidos', fontsize=12, color='#fb8500')
plt.xticks(rotation=45, fontsize=10, color='#219ebc')
plt.yticks(fontsize=10, color='#219ebc')

ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)

ax.yaxis.grid(False, linestyle='', alpha=0.7)

# Mostrar la gráfica
plt.tight_layout()
plt.show()

# Pedidos con Retraso
print(f"---------- EJERCICIO 3 - Consulta 3: Pedidos con Retraso ----------")

consulta_pedidos_retrasados = select(
    orders.c.order_id,
    orders.c.customer_id,
    orders.c.ship_via,
    orders.c.order_date,
    orders.c.required_date,
    orders.c.shipped_date,
    customers.c.company_name.label('customer_company'),
    customers.c.contact_name.label('customer_contact'),
    customers.c.city.label('customer_city'),
    customers.c.country.label('customer_country'),
    shippers.c.company_name.label('shipper_company')
).where(orders.c.customer_id == customers.c.customer_id).where(orders.c.ship_via == shippers.c.shipper_id)

df_pedidos_retrasados = pd.read_sql(consulta_pedidos_retrasados, engine)

df_pedidos_retrasados['order_date'] = pd.to_datetime(df_pedidos_retrasados['order_date'])
df_pedidos_retrasados['required_date'] = pd.to_datetime(df_pedidos_retrasados['required_date'])
df_pedidos_retrasados['shipped_date'] = pd.to_datetime(df_pedidos_retrasados['shipped_date'])

df_pedidos_retrasados['dias_retraso'] = (df_pedidos_retrasados['shipped_date'] - df_pedidos_retrasados['required_date']).dt.days
df_pedidos_retrasados['dias_retraso'] = df_pedidos_retrasados['dias_retraso'].fillna(-1)

df_pedidos_retrasados['estado'] = df_pedidos_retrasados['dias_retraso'].apply(lambda x: 'A tiempo' if x <= 0 else ('Retraso' if x > 0 else 'No registrado'))

df_retrasos_compania = df_pedidos_retrasados[['shipper_company', 'dias_retraso']].copy()
df_retrasos_compania = df_retrasos_compania[df_retrasos_compania['dias_retraso'] != -1]

# Crear la figura y los ejes
plt.figure(figsize=(12, 6), facecolor='#ffffff')
ax = plt.gca()
ax.set_facecolor('none')

df_retrasos_compania.boxplot(by='shipper_company', column='dias_retraso', grid=False, ax=ax, patch_artist=True,
                                 boxprops=dict(facecolor='#219ebc', color='#219ebc'),
                                 medianprops=dict(color='#fb8500'),
                                 whiskerprops=dict(color='#219ebc'),
                                 capprops=dict(color='#219ebc'),
                                 flierprops=dict(color='#fb8500', markeredgecolor='#fb8500'))

# Personalizar el gráfico
plt.title('Retrasos en los Pedidos por Compañía de Transporte', fontsize=16, color='#fb8500')
plt.suptitle('')
plt.xlabel('Compañía de Transporte', fontsize=12, color='#fb8500')
plt.ylabel('Días de Retraso', fontsize=12, color='#fb8500')
plt.xticks(fontsize=10, color='#219ebc')
plt.yticks(fontsize=10, color='#219ebc')

plt.grid(False)

# Mostrar la gráfica
plt.tight_layout()
plt.show()

# Media Precio Pedidos
print(f"---------- EJERCICIO 3 - Consulta 4: Media Precio Pedidos ----------")

consulta_pedidos_detalles_clientes = select(
    orders.c.order_id,
    orders.c.customer_id,
    customers.c.country.label('customer_country'),
    (order_details.c.unit_price * order_details.c.quantity * (1 - order_details.c.discount)).label('total_price')
).where(orders.c.order_id == order_details.c.order_id).where(orders.c.customer_id == customers.c.customer_id)

df_pedidos_detalles_clientes = pd.read_sql(consulta_pedidos_detalles_clientes, engine)

precio_medio_pedido_pais = df_pedidos_detalles_clientes.groupby('customer_country')['total_price'].mean().reset_index()
precio_medio_pedido_pais = precio_medio_pedido_pais.sort_values(by='total_price', ascending=False)

# Crear la figura y los ejes
plt.figure(figsize=(14, 8), facecolor='#ffffff')
ax = plt.gca()
ax.set_facecolor('none')

colors = plt.cm.Blues(precio_medio_pedido_pais['total_price'] / max(precio_medio_pedido_pais['total_price']))

bars = plt.bar(precio_medio_pedido_pais['customer_country'], precio_medio_pedido_pais['total_price'], color=colors)

# Añadir etiquetas y personalizar el gráfico
plt.title('Distribución Media del Precio del Pedido por País de Procedencia del Cliente', fontsize=16, color='#ffb703')
plt.xlabel('País de Procedencia del Cliente', fontsize=12, color='#ffb703')
plt.ylabel('Precio Medio del Pedido', fontsize=12, color='#ffb703')
plt.xticks(rotation=90, fontsize=10, color='#219ebc')
plt.yticks(fontsize=10, color='#219ebc')

# Añadir etiquetas de valor encima de las barras
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom', fontsize=10, color='#219ebc')

# Mostrar la gráfica
plt.tight_layout()
plt.show()

# Clientes sin Pedidos
print(f"---------- EJERCICIO 3 - Consulta 5: Clientes sin Pedidos----------")

consulta_clientes = select(customers.c.customer_id)
consulta_clientes_con_pedidos = select(orders.c.customer_id).distinct()
resultados_clientes = session.execute(consulta_clientes).fetchall()
resultados_clientes_con_pedidos = session.execute(consulta_clientes_con_pedidos).fetchall()

clientes_totales = [cliente[0] for cliente in resultados_clientes]
clientes_con_pedidos = [cliente[0] for cliente in resultados_clientes_con_pedidos]

clientes_sin_pedidos = set(clientes_totales) - set(clientes_con_pedidos)
porcentaje_clientes_sin_pedidos = (len(clientes_sin_pedidos) / len(clientes_totales)) * 100

print(f"Número total de clientes: {len(clientes_totales)}")
print()
print(f"Número de clientes con pedidos: {len(clientes_con_pedidos)}")
print()
print(f"Número de clientes sin pedidos: {len(clientes_sin_pedidos)}")
print()
print(f"Porcentaje de clientes sin pedidos: {porcentaje_clientes_sin_pedidos:.2f}%")
print()

# Productos Demandados
print(f"---------- EJERCICIO 3 - Consulta6: Productos Demandados----------")

consulta_productos_demandados = select(
    products.c.product_id,
    products.c.product_name,
    products.c.units_in_stock,
    func.sum(order_details.c.quantity).label('unidades_pedidas')
).join(order_details, products.c.product_id == order_details.c.product_id).group_by(
    products.c.product_id,
    products.c.product_name,
    products.c.units_in_stock
).order_by(func.sum(order_details.c.quantity).desc())

consulta_reestock_urgente = select(
    products.c.product_id,
    products.c.product_name,
    products.c.units_in_stock
).where(
    products.c.units_in_stock <= 20
).where(
    ~order_details.select().where(
        order_details.c.product_id == products.c.product_id
    ).exists()
)

with engine.connect() as conn:
    # Consulta de productos más demandados
    df_productos_demandados = pd.read_sql(consulta_productos_demandados, conn)

    # Consulta de productos para reestock urgente
    df_reestock_urgente = pd.read_sql(consulta_reestock_urgente, conn)

# Mostrar productos más demandados
print("Productos más demandados por unidades pedidas:")
print(df_productos_demandados.head())
print()

# Mostrar productos para reestock urgente
print("Productos que necesitan reestock urgente (20 o menos unidades en stock y ninguna unidad pedida):")
print(df_reestock_urgente.head())
print()

# Gráfico de barras para productos más demandados
fig, ax = plt.subplots(figsize=(20, 6))
ax.bar(df_productos_demandados['product_name'], df_productos_demandados['unidades_pedidas'], color='#219ebc')

# Personalización del gráfico
ax.set_title('Productos más demandados por unidades pedidas', fontsize=16, color='#fb8500')
ax.set_xlabel('Productos', fontsize=12, color='#fb8500')
ax.set_ylabel('Unidades Pedidas', fontsize=12, color='#fb8500')
ax.tick_params(axis='x', rotation=90, labelsize=10, color='#219ebc')
ax.tick_params(axis='y', labelsize=10, color='#219ebc')
plt.xticks(rotation=90, fontsize=10, color='#219ebc')
plt.yticks(fontsize=10, color='#219ebc')

plt.tight_layout()
plt.show()

# Gráfico de barras para productos que necesitan reestock urgente
fig, ax = plt.subplots(figsize=(20, 6))
ax.bar(df_reestock_urgente['product_name'], df_reestock_urgente['units_in_stock'], color='#fb8500')

# Personalización del gráfico
ax.set_title('Productos que necesitan reestock urgente (20 o menos unidades en stock y ninguna unidad pedida)', fontsize=16, color='#fb8500')
ax.set_xlabel('Productos', fontsize=12, color='#fb8500')
ax.set_ylabel('Unidades en Stock', fontsize=12, color='#fb8500')
ax.tick_params(axis='x', rotation=45, labelsize=10, color='#219ebc')
ax.tick_params(axis='y', labelsize=10, color='#219ebc')
plt.xticks(rotation=90, fontsize=10, color='#219ebc')
plt.yticks(fontsize=10, color='#219ebc')

plt.tight_layout()
plt.show()


#------------------------------Ejercicio 4. Queries Avanzadas------------------------------
print(f"---------- EJERCICIO 4 - Queries Avanzadas ----------")
print()

# Consulta 1: Fecha Ultimo Pedido Producto Categoria
print(f"---------- EJERCICIO 4 - Consulta 1: Fecha Ultimo Pedido Producto Categoria ----------")

consulta_ultima_fecha_por_categoria = (
    session.query(
        categories.c.category_name,
        func.max(orders.c.order_date).label('ultima_fecha_pedido')
    )
    .join(products, products.c.category_id == categories.c.category_id)
    .join(order_details, products.c.product_id == order_details.c.product_id)
    .join(orders, order_details.c.order_id == orders.c.order_id)
    .group_by(categories.c.category_name)
    .all()
)

for resultado in consulta_ultima_fecha_por_categoria:
    print(f"Categoría: {resultado.category_name} - Fecha Ultimo Pedido: {resultado.ultima_fecha_pedido}")
print()

# Consulta 2: Producto no Vendido Precio Original
print(f"---------- EJERCICIO 4 - Consulta 2: Producto no Vendido Precio Original ----------")

consulta_productos_no_vendidos = (
    select(products.c.product_id, products.c.product_name)
    .select_from(products)
    .outerjoin(order_details, products.c.product_id == order_details.c.product_id)
    .where(order_details.c.product_id == None)  # Filtro para productos sin pedidos
)

resultados_productos_no_vendidos = session.execute(consulta_productos_no_vendidos).fetchall()

if resultados_productos_no_vendidos:
    print("Productos que nunca se han vendido por su precio original:")
    for producto in resultados_productos_no_vendidos:
        print(f"ID: {producto.product_id}, Nombre: {producto.product_name}")
else:
    print("No se encontraron productos que nunca se hayan vendido por su precio original.")
print()

# Consulta 3: Producto Categoria "Confections"
print(f"---------- EJERCICIO 4 - Consulta 3: Producto Categoria Confections ----------")

consulta_productos_confections = (
    select(products.c.product_id, products.c.product_name, categories.c.category_id)
    .select_from(products.join(categories, products.c.category_id == categories.c.category_id))
    .where(categories.c.category_name == 'Confections')
)
resultados_productos_confections = session.execute(consulta_productos_confections).fetchall()

if resultados_productos_confections:
    print("Productos de la categoría 'Confections':")
    for producto in resultados_productos_confections:
        print(f"ID Producto: {producto.product_id}, Nombre: {producto.product_name}, ID Categoría: {producto.category_id}")
else:
    print("No se encontraron productos en la categoría 'Confections'.")
print()

# Consulta 4: Proveedor con Productos Discontinuados"
print(f"---------- Consulta 4: Proveedor con Productos Discontinuados ----------")

fecha_limite = func.current_date() - timedelta(days=90)

subquery_productos_descontinuados = (
    select(products.c.product_id)
    .where(~products.c.product_id.in_(
        select(order_details.c.product_id)
        .select_from(order_details.join(orders, order_details.c.order_id == orders.c.order_id))
        .where(orders.c.shipped_date >= fecha_limite)
    ))
)
consulta_proveedores_descontinuados = (
    select(suppliers.c.supplier_id, suppliers.c.company_name)
    .select_from(
        suppliers.outerjoin(products, products.c.supplier_id == suppliers.c.supplier_id)
    )
    .where(products.c.product_id.in_(subquery_productos_descontinuados))
    .group_by(suppliers.c.supplier_id, suppliers.c.company_name)
    .having(func.count(products.c.product_id) == 0)
)

resultados_proveedores_descontinuados = session.execute(consulta_proveedores_descontinuados).fetchall()

if resultados_proveedores_descontinuados:
    print("Proveedores cuyos productos están todos descontinuados:")
    for proveedor in resultados_proveedores_descontinuados:
        print(f"ID Proveedor: {proveedor.supplier_id}, Nombre Empresa: {proveedor.company_name}")
else:
    print("No se encontraron proveedores cuyos productos estén todos descontinuados.")
print()

# Consulta 5: Clientes con Pedidos de Chai mayor 30 Unidades"
print(f"---------- Consulta 5: Clientes con Pedidos de Chai mayor 30 Unidades ----------")

subquery_chai_units = (
    select(
        order_details.c.order_id,
        func.sum(order_details.c.quantity).label('total_chai_quantity')
    )
    .select_from(order_details.join(products, order_details.c.product_id == products.c.product_id))
    .where(products.c.product_name == 'Chai')
    .group_by(order_details.c.order_id)
    .having(func.sum(order_details.c.quantity) > 30)
    .alias('subquery_chai_units')
)

consulta_clientes_chai = (
    select(customers.c.customer_id, customers.c.company_name)
    .select_from(customers.join(orders, customers.c.customer_id == orders.c.customer_id))
    .where(orders.c.order_id == subquery_chai_units.c.order_id)
)

resultados_clientes_chai = session.execute(consulta_clientes_chai).fetchall()

if resultados_clientes_chai:
    print("Clientes que compraron más de 30 unidades de 'Chai' en un único pedido:")
    for cliente in resultados_clientes_chai:
        print(f"ID Cliente: {cliente.customer_id}, Nombre Empresa: {cliente.company_name}")
else:
    print("No se encontraron clientes que cumplan con el criterio.")
print()

# Consulta 6: Clientes cuya Carga en los Pedidos sea superior a 1.000"
print(f"---------- Consulta 6: Clientes cuya Carga en los Pedidos sea superior a 1.000 ----------")

subquery_total_cargas = (
    select(
        orders.c.customer_id,
        func.sum(order_details.c.unit_price * order_details.c.quantity).label('total_carga')
    )
    .select_from(orders.join(order_details, orders.c.order_id == order_details.c.order_id))
    .group_by(orders.c.customer_id)
    .having(func.sum(order_details.c.unit_price * order_details.c.quantity) > 1000)
    .alias('subquery_total_cargas')
)

consulta_clientes_carga = (
    select(customers.c.customer_id, customers.c.company_name)
    .select_from(customers.join(subquery_total_cargas, customers.c.customer_id == subquery_total_cargas.c.customer_id))
)

resultados_clientes_carga = session.execute(consulta_clientes_carga).fetchall()

if resultados_clientes_carga:
    print("Clientes cuya suma total de carga en los pedidos es mayor de 1000:")
    for cliente in resultados_clientes_carga:
        print(f"ID Cliente: {cliente.customer_id}, Nombre Empresa: {cliente.company_name}")
else:
    print("No se encontraron clientes cuya suma total de carga en los pedidos sea mayor de 1000.")
print()

# Consulta 7: Ciudades con mas de 5 Empleados"
print(f"---------- Consulta 7: Ciudades con mas de 5 Empleados ----------")

consulta_ciudades_con_empleados = (
    select(employees.c.city)
    .group_by(employees.c.city)
    .having(func.count(employees.c.employee_id) >= 5)
)

resultados_ciudades = session.execute(consulta_ciudades_con_empleados).fetchall()

if resultados_ciudades:
    print("Ciudades con 5 o más empleados:")
    for ciudad in resultados_ciudades:
        print(ciudad.city)
else:
    print("No se encontraron ciudades con 5 o más empleados.")
