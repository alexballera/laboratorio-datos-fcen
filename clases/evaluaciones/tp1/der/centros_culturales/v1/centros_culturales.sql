CREATE TABLE [provincia] (
  [id] integer PRIMARY KEY,
  [nombre] nvarchar(255)
)
GO

CREATE TABLE [departamento] (
  [id] integer PRIMARY KEY,
  [nombre] nvarchar(255)
)
GO

CREATE TABLE [localidad] (
  [id] integer PRIMARY KEY,
  [nombre] nvarchar(255)
)
GO

CREATE TABLE [categoria] (
  [id] integer PRIMARY KEY,
  [categoria] nvarchar(255)
)
GO

CREATE TABLE [tipo_lat_long] (
  [id] integer PRIMARY KEY,
  [tipo] nvarchar(255)
)
GO

CREATE TABLE [domicilio] (
  [id] integer PRIMARY KEY,
  [direccion] nvarchar(255),
  [piso] nvarchar(255),
  [cp] integer
)
GO

CREATE TABLE [contacto] (
  [id] integer PRIMARY KEY,
  [cod_area] integer,
  [telefono] integer
)
GO

CREATE TABLE [centros_culturales] (
  [cod_loc] integer,
  [id_localidad] integer,
  [id_categoria] integer,
  [id_domicilio] integer,
  [id_contacto] integer,
  [nombre] nvarchar(255),
  [observaciones] nvarchar(255),
  [web] nvarchar(255),
  [info_adicional] nvarchar(255),
  [latitud] float,
  [longitud] float,
  [id_tipo_lat_long] integer,
  [fuente] nvarchar(255),
  [año_inicio] timestamp,
  [capacidad] integer,
  [actualizacion] integer,
  PRIMARY KEY ([nombre], [latitud], [longitud])
)
GO

ALTER TABLE [contacto] ADD FOREIGN KEY ([id]) REFERENCES [centros_culturales] ([id_contacto])
GO

ALTER TABLE [categoria] ADD FOREIGN KEY ([id]) REFERENCES [centros_culturales] ([id_categoria])
GO

ALTER TABLE [localidad] ADD FOREIGN KEY ([id]) REFERENCES [centros_culturales] ([id_localidad])
GO

ALTER TABLE [tipo_lat_long] ADD FOREIGN KEY ([id]) REFERENCES [centros_culturales] ([id_tipo_lat_long])
GO

ALTER TABLE [domicilio] ADD FOREIGN KEY ([id]) REFERENCES [centros_culturales] ([id_domicilio])
GO

ALTER TABLE [provincia] ADD FOREIGN KEY ([id]) REFERENCES [departamento] ([id])
GO

ALTER TABLE [departamento] ADD FOREIGN KEY ([id]) REFERENCES [localidad] ([id])
GO
