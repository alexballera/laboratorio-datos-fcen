CREATE TABLE `localizaciones_departamento` (
  `numero_departamento` integer,
  `ubicacion_departamento` varchar(255),
  PRIMARY KEY (`numero_departamento`, `ubicacion_departamento`)
);

CREATE TABLE `departamento` (
  `nombre_departamento` varchar(255) PRIMARY KEY,
  `numero_departamento` integer,
  `fecha_ingreso_director` timestamp,
  `dni_director` integer
);

CREATE TABLE `empleado` (
  `nombre` varchar(255),
  `apellido1` varchar(255),
  `apellido2` varchar(255),
  `dni` integer PRIMARY KEY,
  `super_dni` integer,
  `fecha_nac` timestamp,
  `direccion` varchar(255),
  `sexo` varchar(255),
  `sueldo` integer,
  `dno` integer
);

CREATE TABLE `proyecto` (
  `nombre_proyecto` varchar(255) PRIMARY KEY,
  `numero_proyecto` integer,
  `ubicacion_proyecto` varchar(255),
  `numero_departamento_proyecto` integer
);

CREATE TABLE `trabaja_en` (
  `dni_empleado` integer,
  `numero_proyecto` integer,
  `horas` integer,
  PRIMARY KEY (`dni_empleado`, `numero_proyecto`)
);

ALTER TABLE `empleado` ADD FOREIGN KEY (`dni`) REFERENCES `empleado` (`super_dni`);

ALTER TABLE `empleado` ADD FOREIGN KEY (`dno`) REFERENCES `departamento` (`numero_departamento`);

ALTER TABLE `departamento` ADD FOREIGN KEY (`dni_director`) REFERENCES `empleado` (`dni`);

ALTER TABLE `localizaciones_departamento` ADD FOREIGN KEY (`numero_departamento`) REFERENCES `departamento` (`numero_departamento`);

ALTER TABLE `proyecto` ADD FOREIGN KEY (`numero_departamento_proyecto`) REFERENCES `departamento` (`numero_departamento`);

ALTER TABLE `trabaja_en` ADD FOREIGN KEY (`dni_empleado`) REFERENCES `empleado` (`dni`);

ALTER TABLE `trabaja_en` ADD FOREIGN KEY (`numero_proyecto`) REFERENCES `proyecto` (`nombre_proyecto`);
