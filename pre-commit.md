# llvm-project
### .clang-format
```
BasedOnStyle: LLVM
LineEnding: LF
```
### .clang-tidy
```
HeaderFilterRegex: ''
Checks: >
  -*,
  clang-diagnostic-*,
  llvm-*,
  misc-*,
  -misc-const-correctness,
  -misc-include-cleaner,
  -misc-no-recursion,
  -misc-non-private-member-variables-in-classes,
  -misc-unused-parameters,
  -misc-use-anonymous-namespace,
  readability-identifier-naming

CheckOptions:
  - key:             readability-identifier-naming.ClassCase
    value:           CamelCase
  - key:             readability-identifier-naming.EnumCase
    value:           CamelCase
  - key:             readability-identifier-naming.FunctionCase
    value:           camelBack
  # Exclude from scanning as this is an exported symbol used for fuzzing
  # throughout the code base.
  - key:             readability-identifier-naming.FunctionIgnoredRegexp
    value:           "LLVMFuzzerTestOneInput"
  - key:             readability-identifier-naming.MemberCase
    value:           CamelCase
  - key:             readability-identifier-naming.ParameterCase
    value:           CamelCase
  - key:             readability-identifier-naming.UnionCase
    value:           CamelCase
  - key:             readability-identifier-naming.VariableCase
    value:           CamelCase
  - key:             readability-identifier-naming.IgnoreMainLikeFunctions
    value:           1
  - key:             readability-redundant-member-init.IgnoreBaseInCopyConstructors
    value:           1
  - key:             modernize-use-default-member-init.UseAssignment
    value:           1
```

# rocksdb
### /.clang-format
```
# Complete list of style options can be found at: 
# http://clang.llvm.org/docs/ClangFormatStyleOptions.html
---
BasedOnStyle: Google
...
```

### /.clang-tidy
```
# When making changes, verify the output of:
#   clang-tidy -list-checks
---
Checks: "-*,\
  bugprone-argument-comment,\
  bugprone-dangling-handle,\
  bugprone-fold-init-type,\
  bugprone-forward-declaration-namespace,\
  bugprone-forwarding-reference-overload,\
  bugprone-shadow,\
  bugprone-sizeof-*,\
  bugprone-string-constructor,\
  bugprone-undefined-memory-manipulation,\
  bugprone-unused-return-value,\
  bugprone-use-after-move,\
  cert-env33-c,\
  cert-err58-cpp,\
  cert-msc30-c,\
  cert-msc50-cpp,\
  clang-analyzer-*,\
  clang-diagnostic-*,\
  -clang-diagnostic-missing-designated-field-initializers,\
  concurrency-mt-unsafe,\
  cppcoreguidelines-avoid-non-const-global-variables,\
  cppcoreguidelines-missing-std-forward,\
  cppcoreguidelines-pro-type-member-init,\
  cppcoreguidelines-special-member-functions,\
  cppcoreguidelines-virtual-class-destructor,\
  google-build-using-namespace,\
  google-explicit-constructor,\
  google-readability-avoid-underscore-in-googletest-name,\
  misc-definitions-in-headers,\
  misc-redundant-expression,\
  modernize-make-shared,\
  modernize-use-emplace,\
  modernize-use-noexcept,\
  modernize-use-override,\
  modernize-use-using,\
  performance-faster-string-find,\
  performance-for-range-copy,\
  performance-implicit-conversion-in-loop,\
  performance-inefficient-algorithm,\
  performance-inefficient-string-concatenation,\
  performance-inefficient-vector-operation,\
  performance-move-const-arg,\
  performance-move-constructor-init,\
  performance-no-automatic-move,\
  performance-no-int-to-ptr,\
  performance-noexcept-move-constructor,\
  performance-noexcept-swap,\
  performance-trivially-destructible,\
  performance-type-promotion-in-math-fn,\
  performance-unnecessary-copy-initialization,\
  performance-unnecessary-value-param,\
  readability-braces-around-statements,\
  readability-duplicate-include,\
  readability-isolate-declaration,\
  readability-operators-representation,\
  readability-redundant-string-init"

WarningsAsErrors: "bugprone-use-after-move"

CheckOptions:
- key: bugprone-easily-swappable-parameters.MinimumLength
  value: 4
- key: cppcoreguidelines-avoid-non-const-global-variables.AllowThreadLocal
  value: true
- key: cppcoreguidelines-special-member-functions.AllowSoleDefaultDtor
  value: true
- key: cppcoreguidelines-special-member-functions.AllowImplicitlyDeletedCopyOrMove
  value: true
- key: modernize-use-using.IgnoreExternC
  value: true
- key: performance-move-const-arg.CheckTriviallyCopyableMove
  value: false
- key: performance-unnecessary-value-param.AllowedTypes
  value: '[Pp]ointer$;[Pp]tr$;[Rr]ef(erence)?$'
- key: performance-unnecessary-copy-initialization.AllowedTypes
  value: '[Pp]ointer$;[Pp]tr$;[Rr]ef(erence)?$'
- key: readability-operators-representation.BinaryOperators
  value: '&&;&=;&;|;~;!;!=;||;|=;^;^='
- key: readability-redundant-string-init.StringNames
  value: '::std::basic_string'
- key: readability-named-parameter.InsertPlainNamesInForwardDecls
  value: true
...
```

# MindIE-LLM
### /.clang-format
```
---
BasedOnStyle: LLVM
---
Language: Cpp
AccessModifierOffset: -4
# AlignAfterOpenBracket: Align
# AlignConsecutiveMacros: false
# AlignConsecutiveAssignments: false
# AlignConsecutiveDeclarations: false
# AlignEscapedNewlines: Left
# AlignOperands: true
# AlignTrailingComments: true
# AllowAllArgumentsOnNextLine: true
AllowAllConstructorInitializersOnNextLine: false
# AllowAllParametersOfDeclarationOnNextLine: true
# AllowShortBlocksOnASingleLine: Never
AllowShortCaseLabelsOnASingleLine: true
# AllowShortFunctionsOnASingleLine: All
# AllowShortLambdasOnASingleLine: All
# AllowShortIfStatementsOnASingleLine: WithoutElse
# AllowShortLoopsOnASingleLine: true
# AlwaysBreakAfterDefinitionReturnType: None
# AlwaysBreakAfterReturnType: None
AlwaysBreakBeforeMultilineStrings: false
# AlwaysBreakTemplateDeclarations: Yes
# BinPackArguments: true
# BinPackParameters: true
BraceWrapping:
  # AfterCaseLabel: false
  AfterClass: false
  AfterControlStatement: false
  AfterEnum: false
  AfterFunction: true
  AfterNamespace: false
  AfterObjCDeclaration: false
  AfterStruct: false
  AfterUnion: false
  AfterExternBlock: false
  BeforeCatch: false
  BeforeElse: false
  IndentBraces: false
  SplitEmptyFunction: true
  SplitEmptyRecord: true
  SplitEmptyNamespace: true
# BreakBeforeBinaryOperators: None
BreakBeforeBraces: Custom
# BreakBeforeInheritanceComma: false
# BreakInheritanceList: BeforeColon
# BreakBeforeTernaryOperators: true
# BreakConstructorInitializersBeforeComma: false
# BreakConstructorInitializers: BeforeColon
# BreakAfterJavaFieldAnnotations: false
# BreakStringLiterals: true
ColumnLimit: 120
CommentPragmas: "^ NOLINT:"
# CompactNamespaces: false
# ConstructorInitializerAllOnOneLineOrOnePerLine: true
# ConstructorInitializerIndentWidth: 4
# ContinuationIndentWidth: 4
# Cpp11BracedListStyle: true
# DeriveLineEnding: true
# DerivePointerAlignment: true
# DisableFormat: false
# ExperimentalAutoDetectBinPacking: false
# FixNamespaceComments: true
# ForEachMacros:
#   - foreach
#   - Q_FOREACH
#   - BOOST_FOREACH
# IncludeBlocks: Regroup
# IncludeCategories:
#   - Regex: '^<ext/.*\.h>'
#     Priority: 2
#     SortPriority: 0
#   - Regex: '^<.*\.h>'
#     Priority: 1
#     SortPriority: 0
#   - Regex: "^<.*"
#     Priority: 2
#     SortPriority: 0
#   - Regex: ".*"
#     Priority: 3
#     SortPriority: 0
# IncludeIsMainRegex: "([-_](test|unittest))?$"
# IncludeIsMainSourceRegex: ""
IndentCaseLabels: true
# IndentGotoLabels: true
# IndentPPDirectives: None
IndentWidth: 4
# IndentWrappedFunctionNames: false
# JavaScriptQuotes: Leave
# JavaScriptWrapImports: true
# KeepEmptyLinesAtTheStartOfBlocks: false
# MacroBlockBegin: ""
# MacroBlockEnd: ""
# MaxEmptyLinesToKeep: 1
# NamespaceIndentation: None
# ObjCBinPackProtocolList: Never
# ObjCBlockIndentWidth: 2
# ObjCSpaceAfterProperty: false
# ObjCSpaceBeforeProtocolList: true
# PenaltyBreakAssignment: 2
# PenaltyBreakBeforeFirstCallParameter: 1
# PenaltyBreakComment: 300
# PenaltyBreakFirstLessLess: 120
# PenaltyBreakString: 1000
# PenaltyBreakTemplateDeclaration: 10
# PenaltyExcessCharacter: 1000000
# PenaltyReturnTypeOnItsOwnLine: 200
PointerAlignment: Right
# RawStringFormats:
#   - Language: Cpp
#     Delimiters:
#       - cc
#       - CC
#       - cpp
#       - Cpp
#       - CPP
#       - "c++"
#       - "C++"
#     CanonicalDelimiter: ""
#     BasedOnStyle: google
#   - Language: TextProto
#     Delimiters:
#       - pb
#       - PB
#       - proto
#       - PROTO
#     EnclosingFunctions:
#       - EqualsProto
#       - EquivToProto
#       - PARSE_PARTIAL_TEXT_PROTO
#       - PARSE_TEST_PROTO
#       - PARSE_TEXT_PROTO
#       - ParseTextOrDie
#       - ParseTextProtoOrDie
#     CanonicalDelimiter: ""
#     BasedOnStyle: google
# ReflowComments: true
SortIncludes: false
SortUsingDeclarations: false
# SpaceAfterCStyleCast: false
# SpaceAfterLogicalNot: false
# SpaceAfterTemplateKeyword: true
# SpaceBeforeAssignmentOperators: true
# SpaceBeforeCpp11BracedList: false
# SpaceBeforeCtorInitializerColon: true
# SpaceBeforeInheritanceColon: true
# SpaceBeforeParens: ControlStatements
# SpaceBeforeRangeBasedForLoopColon: true
# SpaceInEmptyBlock: false
# SpaceInEmptyParentheses: false
SpacesBeforeTrailingComments: 1
# SpacesInAngles: false
# SpacesInConditionalStatement: false
SpacesInContainerLiterals: false
# SpacesInCStyleCastParentheses: false
# SpacesInParentheses: false
# SpacesInSquareBrackets: false
# SpaceBeforeSquareBrackets: false
Standard: Cpp11
# StatementMacros:
#   - Q_UNUSED
#   - QT_REQUIRE_VERSION
TabWidth: 4
# UseCRLF: false
UseTab: Never
```