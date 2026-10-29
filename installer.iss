; ============================================================
; Inno Setup Script for WebMind AI Bridge
;
; Generates a lightweight, user-friendly Windows installer:
; - Installs to %LOCALAPPDATA%\Programs\WebMind (no Admin rights required)
; - Creates Desktop and Start Menu shortcuts
; - Clean uninstaller that leaves user profiles intact
; ============================================================

#define MyAppName "WebMind"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Vaibhav Upadhyaya"
#define MyAppURL "https://github.com/Vaibhav-Upadhyaya-codes/WebMind"
#define MyAppExeName "WebMind.bat"

[Setup]
AppId={{8B19E5D2-94C3-4831-9F3E-656C9321DE77}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=Output
OutputBaseFilename=WebMind-Setup-v{#MyAppVersion}
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "dist\WebMind\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: shellexec postinstall nowait skipifsilent
