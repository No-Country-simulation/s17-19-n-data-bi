**This repository is part of No Country's Simulation Tech Program in Data-BI.**

💎 **Nombre del Producto**

![logo](https://github.com/No-Country-simulation/s17-19-n-data-bi/blob/fe4b237d7698aac78b2d76bd8743c987ff32db77/Pi.png)

📊 **Rubro**

Orientado a Healthtech: Análisis, consulta y previsión para Farmacéuticas y Clientes.

🔗 **Enlaces Relevantes**

- APP: [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://pharma-insights.streamlit.app)
- MVP: [![Trello](https://img.shields.io/badge/Trello-0079BF?logo=trello&logoColor=white)](https://trello.com/b/nGylF9YE/s17-19-n-data-bi)

🦾 **Propuesta de Valor**

Pharma Insights es una plataforma destinada a proporcionar análisis, predicciones, consultas y conocimiento del sector Farmacéutico refiriéndose a:

- Análisis de Mercado,
- Optimización de la Cadena de Suministro,
- Inteligencia de Marketing y Previsión,
- Modelado Epidemiológico.

El enfoque está dividido según foco de interés para la cadena de Farmacias (proveedor) y para el Cliente (consumidor).

📉 **Flujo del Sistema**

La plataforma tiene dos verticales, con acceso de perfil Farmacéutico o Cliente.

El acceso para Farmacéuticas tiene un login inicial, y luego se presenta la lógica de la aplicación, pudiendo consultar acerca de:

- Verificación de Stock:
   - Se entrena un modelo de machine learning en PyTorch para verificar stock disponible para permitir anticipar la demanda de productos y optimizar la gestión de inventario.
   - Se implementa con GitHub Actions, CI/CD en el contexto de machine learning (MLOps) con el objeto de automatizar el ciclo de vida del desarrollo del modelo, desde el entrenamiento hasta el despliegue y monitoreo continuo.
   - En la predicción de stock, este modelo está diseñado para predecir problemas futuros de stock en un sistema de gestión de inventarios. Re reentrena para anticipar cuándo se debe reabastecer el stock o ajustar las estrategias de ventas. Con el pipeline, cada vez que el modelo se actualice (por ejemplo, con nuevos datos de ventas), se entrena automáticamente y se despliega en producción.
 
- Predicción de Consumo:
   - Esta tiene dos orientaciones, una basada en datos según análisis históricos y predictivos y otra con implementación de IA con enfoque preventivo.

- Marketing Intelligece y Afinidad de Productos:
   - En enfoque está basado en prompts característicos y pertinentes acerca de generar un Sistema de Recomendación de Precios y Combos, como así también de Posibles Demandas de Productos Relacionados, considerando asociatividad entre el consumo de un producto en particular con respecto al consumo de otros afines a este, con implementación de IA.
 
- Productos con Cobertura o Sin Cobertura:
   - Esta funcionalidad está enfocada a informar según perfil terapéutico el estado de cobertura de los distintos medicamentos como así también las marcas alternativas consideradas genéricas.

El acceso para Clientes se presenta directamente con la lógica de la aplicación, pudiendo consultar acerca de:

- Verificación de Stock:
   - Tiene como objetivo informar al cliente si determinada Sucursal cuenta con stock disponible del medicamento que está requiriendo, como así también la predicción futura acerca de su disponibilidad.
 
- Verificación de Cobertura:
   - Esta funcionalidad está enfocada a informar según perfil terapéutico el estado de cobertura de los distintos medicamentos como así también las marcas alternativas consideradas genéricas.
 
- Cuidar su Salud:
   - Esta funcionalidad está enfocada a informar con implementación de IA, campañas según contexto epidemiológico y de prevención vigentes.
 
- Newsletter:
   - Una opción con el objetivo de mantenerse al tanto de ofertas y campañas de salud mediante la solicitud de un correo electrónico.

🤖 **Stack Tech**

- Lenguaje de Programación: ![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
- Base de Datos: ![Excel](https://img.shields.io/badge/Excel-336791?logo=microsoft-excel&logoColor=green)
- IDE: ![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?logo=google-colab&logoColor=white) ![Visual Studio Code](https://img.shields.io/badge/Visual_Studio_Code-007ACC?logo=visual-studio-code&logoColor=white)
- Visualización de datos: ![Power BI](https://img.shields.io/badge/Power_BI-F2C811?logo=power-bi&logoColor=white)
- Creación de aplicación web interactiva: ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
- Gestión del Código Fuente y Desarrollo Colaborativo: ![GitHub](https://img.shields.io/badge/GitHub-100000?logo=github&logoColor=white) ![GitHub Actions](https://img.shields.io/badge/-100000?logo=github&logoColor=white)
- MVP y Gestión de Equipo: ![Trello](https://img.shields.io/badge/Trello-0079BF?logo=trello&logoColor=white)

🧩 **Colaboradores**

- Angel Jaramillo Sulca - Data Engineer **>>** [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/angeljarads/) [![GitHub](https://img.shields.io/badge/GitHub-100000?logo=github&logoColor=white)](https://github.com/Angeljs094)

- Brayan Burgos - Data Analyst **>>** [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/brayan-burgos-49715a183/) [![GitHub](https://img.shields.io/badge/GitHub-100000?logo=github&logoColor=white)](https://github.com/brayan2693)

- Marco Caro - Data Analyst **>>** [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/marco-antonio-caro-22459711b/) [![GitHub](https://img.shields.io/badge/GitHub-100000?logo=github&logoColor=white)](https://github.com/brayan2693)

- Franco Gabriel Iribarne - Data Scientist **>>** [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/franco-gabriel-iribarne-4101a32ab/) [![GitHub](https://img.shields.io/badge/GitHub-100000?logo=github&logoColor=white)](https://github.com/marco11235813)

- Alfonso Palacio - Data Scientist **>>** [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/alfonso-palacio/) [![GitHub](https://img.shields.io/badge/GitHub-100000?logo=github&logoColor=white)](https://github.com/Parzival099)

- Cristhian Franco - Data Scientist **>>** [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/cristhian-franco-b17313285) [![GitHub](https://img.shields.io/badge/GitHub-100000?logo=github&logoColor=white)](https://github.com/CoolHero83)

- Delicia Fedele Boria - Machine Learning **>>** [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/deliciafedeleboria/) [![GitHub](https://img.shields.io/badge/GitHub-100000?logo=github&logoColor=white)](https://github.com/defedeleboria/)

- ASESOR DE CÓDIGO Y DEPLOY PARA MACHINE LEARNING: Alejandro Asor Corrales Gómez **>>** [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/aacg/)
