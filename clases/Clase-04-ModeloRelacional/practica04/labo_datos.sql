CREATE TABLE [localizaciones_departamento] (
  [numero_departamento] integer,
  [ubicacion_departamento] nvarchar(255),
  PRIMARY KEY ([numero_departamento], [ubicacion_departamento])
)
GO

CREATE TABLE [departamento] (
  [nombre_departamento] nvarchar(255) PRIMARY KEY,
  [numero_departamento] integer,
  [fecha_ingreso_director] timestamp,
  [dni_director] integer
)
GO

CREATE TABLE [empleado] (
  [nombre] nvarchar(255),
  [apellido1] nvarchar(255),
  [apellido2] nvarchar(255),
  [dni] integer PRIMARY KEY,
  [super_dni] integer,
  [fecha_nac] timestamp,
  [direccion] nvarchar(255),
  [sexo] nvarchar(255),
  [sueldo] integer,
  [dno] integer
)
GO

CREATE TABLE [proyecto] (
  [nombre_proyecto] nvarchar(255) PRIMARY KEY,
  [numero_proyecto] integer,
  [ubicacion_proyecto] nvarchar(255),
  [numero_departamento_proyecto] integer
)
GO

CREATE TABLE [trabaja_en] (
  [dni_empleado] integer,
  [numero_proyecto] integer,
  [horas] integer,
  PRIMARY KEY ([dni_empleado], [numero_proyecto])
)
GO

ALTER TABLE [empleado] ADD FOREIGN KEY ([dni]) REFERENCES [empleado] ([super_dni])
GO

ALTER TABLE [empleado] ADD FOREIGN KEY ([dno]) REFERENCES [departamento] ([numero_departamento])
GO

ALTER TABLE [departamento] ADD FOREIGN KEY ([dni_director]) REFERENCES [empleado] ([dni])
GO

ALTER TABLE [localizaciones_departamento] ADD FOREIGN KEY ([numero_departamento]) REFERENCES [departamento] ([numero_departamento])
GO

ALTER TABLE [proyecto] ADD FOREIGN KEY ([numero_departamento_proyecto]) REFERENCES [departamento] ([numero_departamento])
GO

ALTER TABLE [trabaja_en] ADD FOREIGN KEY ([dni_empleado]) REFERENCES [empleado] ([dni])
GO

ALTER TABLE [trabaja_en] ADD FOREIGN KEY ([numero_proyecto]) REFERENCES [proyecto] ([nombre_proyecto])
GO
