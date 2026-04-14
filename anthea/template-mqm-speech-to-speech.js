// Copyright 2026 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * This is an MQM template for evaluating Speech-to-Speech translation.
 * It provides play buttons aligned to text segments.
 */
antheaTemplates['MQM-Speech-to-Speech'] = {
  severities: {
    major: {
      display: 'Major severity',
      shortcut: 'M',
      color: '#fca5a5',
      description: 'Major severity errors significantly alter the meaning ' +
                   'of the source speech, or significantly degrade the quality ' +
                   'of the speech.',
    },
    minor: {
      display: 'Minor severity',
      shortcut: 'm',
      color: '#fef08a',
      description: 'Minor severity errors are noticeable but minor flaws in the ' +
                   'translated speech. They do not significantly alter the ' +
                   'meaning of the source speech, and they do not ' +
                   'significantly degrade the quality of the speech.',
    },
  },

  /**
   * @const {string} Template version identifier.
   */
  VERSION: 'v1.00-Apr-10-2026',

  /**
   * @const {boolean} Show two translations when set to true.
   */
  SIDE_BY_SIDE: false,

  /**
   * @const {boolean} Collect per-segment quality scores when set to true. Also
   *    disables splitting segments into sub-paragraphs.
   */
  COLLECT_QUALITY_SCORE: false,

  /**
   * @const {boolean} Only rate the target side, i.e., the translated text.
   */
  TARGET_SIDE_ONLY: false,

  /**
   * @const {number} Allow marking at most these many errors per sentence. If
   *     set to 0, then no limit is imposed.
   */
  MAX_ERRORS: 0,

  /**
   * @const {boolean} Set this to true if the template instructions already
   *     include listings of errors and severities, and you do not want to
   *     auto-append the lists of errors/severities to the instructions.
   */
  SKIP_RATINGS_TABLES: true,

  /**
   * @const {boolean} Set this to true if you want to allow error spans to
   *     start on whitespace.
   */
  ALLOW_SPANS_STARTING_ON_SPACE: true,

  /**
   * @const {boolean} Set this to true if you want to present the error
   *    types/subtypes in a short, flat list (avoid using this if you have
   *    more than 10 error type/subtype combinations).
   */
  FLATTEN_SUBTYPES: false,

  /**
   * @const {boolean} Set this to true if your TSV data for each docsys
   *     includes a web page rendering and bounding box provided as an
   *     annotation on each first segment. The annotation should be a JSON
   *     string that looks like this (exccept that it should not have
   *     newlines):
   * {
   *   "source": {
   *     "url": "http://screenshot-image-url-for-source",
   *     "w": 100, "h": 120, "box": {"x": 5, "y": 7, "w": 90, "h": 30}
   *   },"
   *   "target": {
   *     "url": "http://screenshot-image-url-for-target",
   *     "w": 100, "h": 120, " "box": {"x": 5, "y": 7, "w": 90, "h": 30}
   *   },
   * }
   */
  USE_PAGE_CONTEXT: false,

  errors: {
    accuracy: {
      display: 'Accuracy',
      description: 'The target text does not accurately reflect the source text.',
      subtypes: {
        reinterpretation: {
          display: 'Creative Reinterpretation',
          description: 'The target text reinterprets the source, but preserves its intent within its broader context (the document and its purpose, and the target locale). This can be because the translation adds, removes, or modifies text in accordance with the target locale, or simply makes creative changes that fully preserve the intent of the source text.',
        },
        mistranslation: {
          display: 'Mistranslation',
          description: 'The target text does not accurately represent the source text.',
        },
        gender_mismatch: {
          display: 'Gender Mismatch',
          description: 'The gender is incorrect (incorrect pronouns, noun/adjective endings, etc).',
        },
        untranslated: {
          display: 'Source language fragment',
          description: 'Content that should have been translated has been left untranslated.',
        },
        addition: {
          display: 'Addition',
          description: 'The target text includes information not present in the source.',
        },
        omission: {
          display: 'Omission',
          description: 'Content is missing from the translation that is present in the source.',
          source_side_only: true,
        },
      },
    },
    fluency: {
      display: 'Fluency',
      description: 'Issues related to the form or content of translated text, independent of its relation to the source text; errors in the translated text that prevent it from being understood clearly.',
      subtypes: {
        inconsistency: {
          display: 'Inconsistency',
          description: 'The text shows internal inconsistency (not related to terminology).',
        },
        grammar: {
          display: 'Grammar',
          description: 'Issues related to the grammar or syntax of the text, other than spelling and orthography.',
        },
        register: {
          display: 'Register',
          description: 'The content uses the wrong grammatical register, such as using informal pronouns or verb forms when their formal counterparts are required.',
        },
        spelling: {
          display: 'Spelling',
          description: 'Issues related to spelling or capitalization of words, and incorrect omission/addition of whitespace.',
        },
        breaking: {
          display: 'Text-Breaking',
          description: 'Issues related to missing or unwarranted paragraph breaks or line breaks.',
        },
        punctuation: {
          display: 'Punctuation',
          description: 'Punctuation is used incorrectly (for the locale or style).',
        },
        character_encoding: {
          display: 'Character encoding',
          description: 'Characters are garbled due to incorrect application of an encoding.',
        },
      },
    },
    style: {
      display: 'Style',
      description: 'The text has stylistic problems.',
      subtypes: {
        awkward: {
          display: 'Unnatural or awkward',
          description: 'The text is literal, written in an awkward style, unidiomatic or inappropriate in the context.',
        },
        sentence_structure: {
          display: 'Bad sentence structure',
          description: 'The marked span of text is an unnecessary repetition, or makes the sentence unnecessarily long, or would have been better as a clause in the previous sentence.'
        },
        archaic_word: {
          display: 'Archaic or obscure word choice',
          description: 'An archaic or lesser-known word is used where a more colloquial term would be a better fit.',
        },
      },
    },
    idiom_culture_consistency: {
      display: 'Idiom/culture consistency',
      description: 'The ability to culturally localize idioms, jokes, or culturally specific references on the fly so the target audience reacts exactly as the source audience did.',
    },
    terminology: {
      display: 'Terminology',
      description: 'A term (domain-specific word) is translated with a term other than the one expected for the domain implied by the context.',
      subtypes: {
        inappropriate: {
          display: 'Inappropriate for context',
          description: 'Translation does not adhere to appropriate industry standard terminology or contains terminology that does not fit the context.',
        },
        inconsistent: {
          display: 'Inconsistent',
          description: 'Terminology is used in an inconsistent manner within the text.',
        },
      },
    },
    locale_convention: {
      display: 'Locale convention',
      description: 'The text does not adhere to locale-specific mechanical conventions and violates requirements for the presentation of content in the target locale.',
      subtypes: {
        address: {
          display: 'Address format',
          description: 'Content uses the wrong format for addresses.',
        },
        date: {
          display: 'Date format',
          description: 'A text uses a date format inappropriate for its locale.',
        },
        currency: {
          display: 'Currency format',
          description: 'Content uses the wrong format for currency.',
        },
        telephone: {
          display: 'Telephone format',
          description: 'Content uses the wrong form for telephone numbers.',
        },
        time: {
          display: 'Time format',
          description: 'Content uses the wrong form for time.',
        },
        name: {
          display: 'Name format',
          description: 'Content uses the wrong form for name.',
        },
      },
    },
    source_audio_mismatch: {
      display: 'Source/Audio Mismatches',
      description: 'Mismatches between the source audio (or text) and the output delivery.',
      subtypes: {
        voice_mismatch: {
          display: 'Voice Mismatch',
          description: 'The target speaker appears to be a different person than the source speaker (e.g. incorrect gender, age, or general vocal identity).',
          auto_expand_span: true,
        },
        affective_mismatch: {
          display: 'Affective Mismatch',
          description: 'Overall emotional tone (happy, sad, urgent) does not match the source speaker.',
          auto_expand_span: true,
        },
        volume_mismatch: {
          display: 'Volume Mismatch',
          description: 'The segment volume characteristics do not match the source.',
          auto_expand_span: true,
        },
        accent_bleeding: {
          display: 'Accent Bleeding',
          description: 'Clone adopts target language accent instead of source speaker\'s characteristic accent.',
          auto_expand_span: true,
        },
      },
    },
    audio_voice_quality: {
      display: 'Audio/Voice Quality',
      description: 'Synthesis quality issues and artifacts, capturing both holistic metallic sounds and localized glitches.',
      subtypes: {
        metallic_thin: {
          display: 'Metallic / Thin',
          description: 'The overall voice sounds "tinny" or artificial due to neural artifacts.',
          auto_expand_span: true,
        },
        background_staining: {
          display: 'Background Staining',
          description: 'Environmental noise from the source is "baked" into the voice output.',
          auto_expand_span: true,
        },
        timbre_inconsistency: {
          display: 'Timbre Inconsistency',
          description: 'The vocal "texture" or identity suddenly shifts for a specific word or phrase.',
        },
        breath_irregularity: {
          display: 'Breath Irregularity',
          description: 'Specific synthetic "gasping" or unnatural breath sounds.',
        },
        choppiness: {
          display: 'Choppiness',
          description: 'Audio sounds stitched together; words or sounds are cut off abruptly.',
        },
        jitter_pops: {
          display: 'Jitter & Pops',
          description: 'Audible "clicks" or micro-stutters where audio chunks are joined.',
        },
        echo_repetition: {
          display: 'Echo / Repetition',
          description: 'A word, phrase, or sound is unnaturally repeated.',
        },
        audio_artifacts: {
          display: 'Audio Artifacts',
          description: 'Specific instances of static, high-pitched noises, or non-speech hums.',
        },
      },
    },
    prosody_delivery: {
      display: 'Prosody & Delivery',
      description: 'Rhythmic, intonation, and delivery pacing issues.',
      subtypes: {
        robotic_cadence: {
          display: 'Robotic Cadence',
          description: 'Evenly spaced, monotonous timing across the segment.',
          auto_expand_span: true,
        },
        inappropriate_prosody: {
          display: 'Inappropriate Prosody',
          description: 'Delivery style is inappropriate for the target language or context.',
          auto_expand_span: true,
        },
        emphasis: {
          display: 'Emphasis Error',
          description: 'Stress is placed on the wrong word/syllable, changing the intent.',
        },
        intonation_drift: {
          display: 'Intonation Drift',
          description: 'A localized pitch change that contradicts the sentence type (e.g., rising pitch on a statement).',
        },
        awkward_chunking: {
          display: 'Awkward chunking',
          description: 'Pauses between chunks that make the sentence hard to parse.',
        },
        inappropriate_pauses: {
          display: 'Inappropriate Pauses',
          description: 'Excessive or unnatural silences between specific words.',
        },
      },
    },
    intelligibility: {
      display: 'Intelligibility',
      description: 'Audio-specific errors related to the clarity and correctness of spoken words.',
      subtypes: {
        mispronunciation: {
          display: 'Mispronunciation',
          description: 'Words are phonetically incorrect, garbled, or "slurred."',
        },
        heteronym: {
          display: 'Heteronym Error',
          description: 'Incorrect pronunciation of words spelled the same (e.g., "lead" vs. "lead").',
        },
        truncation: {
          display: 'Truncation',
          description: 'Words or sentence endings are cut off prematurely.',
        },
      },
    },
    other: {
      display: 'Other',
      description: 'Any other issues (please provide a short description when prompted).',
      needs_note: true,
      source_side_ok: true,
    },
    non_translation: {
      display: 'Non-translation!',
      description: 'The whole sentence is completely not a translation of the source. This rare category, when used, overrides any other marked errors for that sentence, and labels the full translated sentence as the error span. Only available after choosing a major error.',
      forced_severity: 'major',
      override_all_errors: true,
    },
    source_error: {
      display: 'Source issue',
      description: 'Any issue in the source.',
      source_side_only: true,
    },
  },

  /**
   * Add instructions about audio playback and new error categories.
   */
  instructions_section_contents: {
    '_style': `
      <style>
        .anthea-mqm-instructions .summary-heading {
          font-weight: bold;
          font-size: 125%;
        }
        .anthea-mqm-instructions th,
        .anthea-mqm-instructions td {
          border: 1px solid gray;
          vertical-align: top;
          padding: 4px;
        }
        .anthea-mqm-instructions td:first-child {
          font-weight: bold;
        }
        .anthea-mqm-instructions table {
          border-collapse: collapse;
        }
        .span-major {
          background-color: #fca5a5;
        }
        .span-minor {
          background-color: #fef08a;
        }
      </style>
      `,
    'Overview': `
      <h2>Overview</h2>
      <p>
        In this project, you will evaluate speech-to-speech translations.
        For each segment, you can listen to the source or target audio by
        clicking the Play button next to the text (or pressing <code>Spacebar</code>).
      </p>
      <p>
        Please listen to the audio while reading the transcript to evaluate
        the translation quality. You will mark both standard translation errors
        and speech-specific quality issues.
      </p>
      <p>
        <b>Important:</b> The primary object of evaluation is the <b>audio</b>. The text transcript is provided to facilitate annotation. The transcript may contain automatic transcription errors (e.g., a word in the text that sounds similar to what was actually spoken). In these cases, you should only mark an error if the <b>audio</b> itself contains an error. Text-only error types (e.g. character encoding) should be marked based on the text as normal.
      </p>
      `,
    'General Guidelines': `
      <h2>General Guidelines</h2>
      <p>
        The standard you should be reviewing the translations against is
        <b>human speech-to-speech translation quality</b>. Report every occurrence where the
        translation falls short of that standard. Remember that the content you
        are reviewing may be machine-generated.
        <b><i>Apply the same standard regardless</i></b>. The translation should
        be:
      </p>
      <ul>
        <li>Linguistically correct</li>
        <li>Accurate</li>
        <li>Intelligible and natural-sounding (correct pronunciation, appropriate prosody, and good voice quality)</li>
        <li>With terminology appropriate in the context</li>
        <li>Consistent</li>
        <li>Faithful in tone and register to the source speaker</li>
        <li>Appropriately transformed for the target context.</li>
      </ul>
      <p>
        Please be mindful of the following:
      </p>
      <ol>
        <li>
          Before you start annotating, please read carefully the definitions
          for severities and error/issue types.
        </li>
        <li>
          Take both the audio and the transcript into account when annotating.
        </li>
        <li>
          If the whole translation of a segment is so bad that all or nearly
          all of it is completely wrong (e.g., completely nonsensical output, severely broken syntax, or unintelligible audio), then apply the “Major”
          severity and pick the error type “Non-translation!”.
        </li>
      </ol>
      `,
    'Navigation': `
      <h2>Navigation</h2>
      <p>
        Each task consists of audio and transcript from a single document alongside its translation.
      </p>
      <ul>
        <li>
          You will go through a document in steps of segments. You can move from one segment to the next (and back) using the arrow keys or using the buttons labeled with left and right arrows.
        </li>
        <li>
          <b>Audio Playback:</b> You must listen to the source and target audio completely before moving to the next segment. The play button will turn <span style="color: green; font-weight: bold;">green</span> when fully played.
        </li>
        <li>
          Press <code>Spacebar</code> or click the play button to <b>pause</b> and <b>resume</b> audio. Paused audio remembers its position.
        </li>
        <li>
          You can use the <code>Tab</code> key to jump between the source side and the translation side. Switching sides with <code>Tab</code> will not affect the other side's playback position or highlighting.
        </li>
        <li>
          <b>Synchronous Highlighting:</b> As the audio plays, the corresponding text tokens will be highlighted.
        </li>
        <li>
          You can also directly click on any previously read/listened word in the text to resume playback from that point.
        </li>
      </ul>
      `,
    'Annotation Process': `
      <h2>Annotation Process</h2>
      <ol>
        <li>Review the translation of each segment against the source by listening to the audio and reading the transcript, following the general guidelines above.</li>
        <li>
          Select the <b>span</b> of words affected by the issue.
          <ul>
            <li>For most errors, click on the word where the issue begins, then on the word where it ends. If it's only one word, click on it twice.</li>
            <li>The marked span should be the minimal contiguous sequence affected by the issue.</li>
            <li><b>Full-Segment Errors:</b> For error categories labeled <b>(Full-Segment)</b> (e.g., Voice Mismatch, Metallic/Thin), the span will automatically expand to cover the full text block regardless of which word you select. Visually, these are shown as a colored border around the text block. Mark each full-segment error only <b>once per segment</b>.</li>
            <li>When the error is an omission, the error span must be selected on the source side.</li>
          </ul>
        </li>
        <li>
          Select the <b>severity</b> of the issue (Major or Minor).
        </li>
        <li>Select the <b>category</b> and <b>subcategory</b> of the error/issue found.</li>
        <li><b>Play from Here:</b> When you click on a word to start marking an error span, a small <b>▶</b> button labeled "Play from here" will appear above the clicked word. Clicking it will begin audio playback from that word's position.</li>
        <li><b>Error Audio:</b> When you mark an error, a mini play button will appear next to the error description on the right, allowing you to replay just the audio corresponding to the marked span.</li>
        <li>After annotating all identified issues in a segment and ensuring audio has been fully played, use the <b>right arrow key</b> (or the <b>button</b>) to go to the next segment.</li>
      </ol>
      `,
    'Speech Error Categories': `
      <h2>Speech Error Categories</h2>
      <p>
        In addition to the standard textual translation errors (Accuracy, Fluency, Style, etc.), this template introduces several new categories for speech evaluation.
      </p>
      <h3>Translation Quality</h3>
      <ul>
        <li><b>Idiom/culture consistency:</b> The ability to culturally localize idioms, jokes, or culturally specific references on the fly so the target audience reacts exactly as the source audience did.</li>
      </ul>
      <h3>Audio and Speech Quality</h3>
      <ul>
        <li>
          <b>Source/Audio Mismatches.</b> Mismatches between the source audio (or text) and the output delivery.
          <details>
            <summary>Subtypes of Source/Audio Mismatches:</summary>
            <ul>
              <li><b>Voice Mismatch</b> <i>(Full-Segment)</i>. The target speaker appears to be a different person than the source speaker (e.g. incorrect gender, age, or general vocal identity).</li>
              <li><b>Affective Mismatch</b> <i>(Full-Segment)</i>. Overall emotional tone (happy, sad, urgent) does not match the source speaker.</li>
              <li><b>Volume Mismatch</b> <i>(Full-Segment)</i>. The segment volume characteristics do not match the source.</li>
              <li><b>Accent Bleeding</b> <i>(Full-Segment)</i>. Clone adopts target language accent instead of source speaker's characteristic accent.</li>
            </ul>
          </details>
        </li>
        <li>
          <b>Audio/Voice Quality.</b> Synthesis quality issues and artifacts, capturing both holistic metallic sounds and localized glitches.
          <details>
            <summary>Subtypes of Audio/Voice Quality:</summary>
            <ul>
              <li><b>Metallic / Thin</b> <i>(Full-Segment)</i>. The overall voice sounds "tinny" or artificial due to neural artifacts.</li>
              <li><b>Background Staining</b> <i>(Full-Segment)</i>. Environmental noise from the source is "baked" into the voice output.</li>
              <li><b>Timbre Inconsistency.</b> The vocal "texture" or identity suddenly shifts for a specific word or phrase.</li>
              <li><b>Breath Irregularity.</b> Specific synthetic "gasping" or unnatural breath sounds.</li>
              <li><b>Choppiness.</b> Audio sounds stitched together; words or sounds are cut off abruptly.</li>
              <li><b>Jitter &amp; Pops.</b> Audible "clicks" or micro-stutters where audio chunks are joined.</li>
              <li><b>Echo / Repetition.</b> A word, phrase, or sound is unnaturally repeated.</li>
              <li><b>Audio Artifacts.</b> Specific instances of static, high-pitched noises, or non-speech hums.</li>
            </ul>
          </details>
        </li>
        <li>
          <b>Prosody &amp; Delivery.</b> Rhythmic, intonation, and delivery pacing issues.
          <details>
            <summary>Subtypes of Prosody &amp; Delivery:</summary>
            <ul>
              <li><b>Robotic Cadence</b> <i>(Full-Segment)</i>. Evenly spaced, monotonous timing across the segment.</li>
              <li><b>Inappropriate Prosody</b> <i>(Full-Segment)</i>. Delivery style is inappropriate for the target language or context.</li>
              <li><b>Emphasis Error.</b> Stress is placed on the wrong word/syllable, changing the intent.</li>
              <li><b>Intonation Drift.</b> A localized pitch change that contradicts the sentence type (e.g., rising pitch on a statement).</li>
              <li><b>Awkward chunking.</b> Pauses between chunks that make the sentence hard to parse.</li>
              <li><b>Inappropriate Pauses.</b> Excessive or unnatural silences between specific words.</li>
            </ul>
          </details>
        </li>
        <li>
          <b>Intelligibility.</b> Audio-specific errors related to the clarity and correctness of spoken words.
          <details>
            <summary>Subtypes of Intelligibility:</summary>
            <ul>
              <li><b>Mispronunciation.</b> Words are phonetically incorrect, garbled, or "slurred."</li>
              <li><b>Heteronym Error.</b> Incorrect pronunciation of words spelled the same (e.g., "lead" vs. "lead").</li>
              <li><b>Truncation.</b> Words or sentence endings are cut off prematurely.</li>
            </ul>
          </details>
        </li>
      </ul>
      `,
  },

  // Default instructions section order.
  instructions_section_order: [
    '_style', 'Overview', 'General Guidelines', 'Navigation',
    'Annotation Process', 'Annotation Tips', 'Severities defined',
    'Error Types and Subtypes defined', 'Speech Error Categories', 'Annotations exemplified in detail',
    'Style &amp; Convention Guidelines', 'Feedback'
  ],
};
