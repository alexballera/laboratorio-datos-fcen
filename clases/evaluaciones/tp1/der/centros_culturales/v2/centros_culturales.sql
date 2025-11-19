CREATE TABLE [provincia] (
  [id] integer PRIMARY KEY,
  [nombre] nvarchar(255)
)
GO

CREATE TABLE [departamento] (
  [id] integer PRIMARY KEY,
  [nombre] nvarchar(255),
  [id_provincia] integer
)
GO

CREATE TABLE [localidad] (
  [id] integer PRIMARY KEY,
  [nombre] nvarchar(255),
  [id_departamento] integer
)
GO

CREATE TABLE [centros_culturales] (
  [nombre] nvarchar(255),
  [latitud] float,
  [longitud] float,
  [id_localidad] integer,
  [correo] string,
  [capacidad] integer,
  PRIMARY KEY ([latitud], [longitud], [nombre])
)
GO

CREATE TABLE [establecimientos_educativos] (
  [cue] nvarchar(255),
  [nombre] nvarchar(255),
  [jurisdiccion] nvarchar(255),
  [inicial_maternal] nvarchar(255),
  [inicial_infantes] nvarchar(255),
  [primaria] nvarchar(255),
  [secundaria] nvarchar(255),
  [secundaria_inet] nvarchar(255),
  PRIMARY KEY ([cue], [nombre])
)
GO

CREATE TABLE [padron_poblacional] (
  [area] nvarchar(255),
  [localidad] nvarchar(255),
  [edad] integer,
  [cantidad] integer,
  PRIMARY KEY ([area], [localidad])
)
GO

ALTER TABLE [departamento] ADD FOREIGN KEY ([id_provincia]) REFERENCES [provincia] ([id])
GO

ALTER TABLE [localidad] ADD FOREIGN KEY ([id_departamento]) REFERENCES [departamento] ([id])
GO

ALTER TABLE [centros_culturales] ADD FOREIGN KEY ([id_localidad]) REFERENCES [localidad] ([id])
GO
