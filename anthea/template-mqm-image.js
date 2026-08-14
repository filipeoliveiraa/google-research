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
 * This is an MQM template for evaluating Visual and Multimodal Machine
 * Translation (e.g., text in images such as signs, storefronts, menus, posters,
 * packaging, and user interfaces).
 *
 * Both source and target sides display an image for visual context alongside
 * text transcripts that raters use to select spans when logging errors found
 * in the target image.
 */
antheaTemplates['MQM-Image'] = {
  severities: {
    major: {
      display: 'Major severity',
      shortcut: 'M',
      color: '#fca5a5',
      description: 'Major severity errors significantly alter the meaning ' +
          'of the source content, severely impair comprehension, or ' +
          'substantially degrade visual/linguistic quality.',
    },
    minor: {
      display: 'Minor severity',
      shortcut: 'm',
      color: '#fef08a',
      description:
          'Minor severity errors are noticeable but minor flaws in the ' +
          'translated text or its visual presentation. They do not ' +
          'significantly alter meaning and do not hinder comprehension.',
    },
  },

  /**
   * @const {string} Template version identifier.
   */
  VERSION: 'v1.00-Aug-11-2026',

  /**
   * @const {boolean} Show two translations when set to true.
   */
  SIDE_BY_SIDE: false,

  /**
   * @const {boolean} Collect per-segment quality scores when set to true.
   */
  COLLECT_QUALITY_SCORE: false,

  /**
   * @const {boolean} Only rate the target side, i.e., the translated text.
   */
  TARGET_SIDE_ONLY: false,

  /**
   * @const {boolean} Show image translation for the source and target,
   * provided via a JSON annotation on the first segment of each docsys that
   * looks like:
   * {"image_translation":
   *   {"source": "https://source-image-url",
   *    "target": "https://target-image-url"}}
   */
  USE_IMAGE_TRANSLATION: true,

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
   *    types/subtypes in a short, flat list.
   */
  FLATTEN_SUBTYPES: false,

  /**
   * @const {boolean} Web page rendering flag.
   */
  USE_PAGE_CONTEXT: false,

  errors: {
    accuracy: {
      display: 'Accuracy',
      description:
          'The translated text in the target image does not accurately ' +
          'reflect the content or intent of the source image.',
      subtypes: {
        addition: {
          display: 'Addition',
          description:
              'The translated text in the target image includes ' +
              'information or content not present in the source image.',
        },
        mistranslation: {
          display: 'Mistranslation',
          description:
              'The translated text in the target image does not ' +
              'accurately convey the meaning of the source image.',
        },
        omission: {
          display: 'Omission',
          description:
              'Content present in the source image is missing from the ' +
              'translation in the target image.',
          source_side_only: true,
        },
        untranslated: {
          display: 'Source language fragment',
          description:
              'Content that should have been translated has been left ' +
              'untranslated in the source language in the target image.',
        },
        positioning: {
          display: 'Positioning',
          description:
              'Translated text in the target image is swapped, placed on ' +
              'the wrong visual element, or dislocated in a way that ' +
              'degrades understanding.',
        },
      },
    },
    fluency: {
      display: 'Fluency',
      description:
          'Issues related to the form, linguistic quality, or visual ' +
          'character rendering of the translated text in the target image, ' +
          'independent of its relation to the source.',
      subtypes: {
        character_rendering: {
          display: 'Character rendering',
          description:
              'Visually corrupted, garbled, illegible, or incorrectly ' +
              'rendered characters, diacritics/accents, or glyphs in the ' +
              'target image (including reversed LTR rendering or ' +
              'disjointed cursive letters in RTL scripts like Arabic).',
        },
        grammar: {
          display: 'Grammar',
          description:
              'Issues related to the grammar or syntax of the text in the ' +
              'target image, other than spelling and orthography.',
        },
        inconsistency: {
          display: 'Inconsistency',
          description:
              'Inconsistent terminology, tone, honorifics, or naming ' +
              'conventions within the translation in the target image.',
        },
        punctuation: {
          display: 'Punctuation',
          description:
              'Punctuation is used incorrectly, missing, or unnecessary ' +
              'for the locale or style in the target image.',
        },
        register: {
          display: 'Register',
          description:
              'The content in the target image uses the wrong grammatical ' +
              'register or an inappropriate level of formality/tone for ' +
              'the context.',
        },
        spelling: {
          display: 'Spelling',
          description:
              'Issues related to spelling, capitalization of words, or ' +
              'incorrect omission/addition of whitespace in the target ' +
              'image.',
        },
      },
    },
    style: {
      display: 'Style',
      description: 'The text in the target image has stylistic problems.',
      subtypes: {
        awkward: {
          display: 'Awkward or unnatural word choice',
          description:
              'The text in the target image is literal, written in an ' +
              'awkward style, unidiomatic, or inappropriate in the ' +
              'visual/situational context.',
        },
        sentence_structure: {
          display: 'Bad sentence structure',
          description:
              'Sentences in the target image are structured in a ' +
              'confusing or awkward way, or contain unnecessary ' +
              'repetition, even if grammatically correct.',
        },
      },
    },
    terminology: {
      display: 'Terminology',
      description:
          'A domain-specific term in the target image is translated with ' +
          'a term other than the one expected for the domain implied by ' +
          'the context.',
      subtypes: {
        inappropriate: {
          display: 'Inappropriate for context',
          description:
              'Translation does not adhere to appropriate industry ' +
              'standard terminology or contains terminology that does ' +
              'not fit the context.',
        },
        inconsistent: {
          display: 'Inconsistent use',
          description:
              'Different translations are used for the same source term ' +
              'within the document.',
        },
      },
    },
    locale_convention: {
      display: 'Locale convention',
      description:
          'The text in the target image does not adhere to ' +
          'locale-specific mechanical conventions and violates ' +
          'requirements for presentation in the target locale.',
      subtypes: {
        address: {
          display: 'Address format',
          description:
              'Content uses the wrong format for postal addresses.',
        },
        date: {
          display: 'Date format',
          description:
              'Content uses a date format inappropriate for the target ' +
              'locale.',
        },
        currency: {
          display: 'Currency format',
          description:
              'Content uses the wrong format for currency (symbol ' +
              'placement, decimal formatting).',
        },
        telephone: {
          display: 'Telephone / fax format',
          description:
              'Content uses the wrong format for telephone or fax ' +
              'numbers.',
        },
        time: {
          display: 'Time format',
          description:
              'Content uses the wrong form for time (e.g., 12-hour vs. ' +
              '24-hour conventions).',
        },
        name: {
          display: 'Name format',
          description:
              'Content uses the wrong form for personal names (e.g., ' +
              'surname ordering).',
        },
        url: {
          display: 'URL format',
          description:
              'Format errors for hyperlinks and web domains (e.g., ' +
              'translating link text directly into the target language, ' +
              'breaking web addresses or domain names).',
        },
      },
    },
    other: {
      display: 'Other',
      description:
          'Any other issues (please provide a short description when ' +
          'prompted).',
      needs_note: true,
      source_side_ok: true,
    },
    non_translation: {
      display: 'Non-translation!',
      description:
          'The whole segment in the target image is completely not a ' +
          'translation of the source (e.g., total gibberish or entirely ' +
          'unrelated output). This rare category overrides any other ' +
          'marked errors for that segment. Only available after choosing ' +
          'a major error.',
      forced_severity: 'major',
      override_all_errors: true,
    },
    source_error: {
      display: 'Source issue',
      description:
          'Any issue in the original source image or source transcript.',
      source_side_only: true,
    },
  },

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
        In this project, you will evaluate the quality of <b>Image
        Translation</b>. In Image Translation, an input
        <b>source image</b> containing text (such as a road sign, storefront,
        poster, menu, product packaging, or user interface) is transformed into
        a <b>target translated image</b> where the source text is replaced by
        translated text in a target language, while preserving visual elements,
        layout, aesthetics, and typography.
      </p>
      <p>
        Each example displays:
      </p>
      <ul>
        <li><b>Source column:</b> The original source image alongside a
            corresponding text transcript of what appears on the image.</li>
        <li><b>Translation column:</b> The translated target image alongside a
            corresponding text transcript of the translated text.</li>
      </ul>
      <p>
        <b>Core Evaluation Principle:</b> The <b>target image is the ground
        truth output under evaluation</b>. You are evaluating the quality,
        accuracy, visual presentation, and fluency of the translation
        <i>as it appears in the target image</i>. The text transcripts below the
        images serve purely as an interactive interface to allow you to select
        word spans when logging errors found in the image. You must
        <b>not</b> annotate errors that are present only in the transcript and
        not in the image itself.
      </p>
      `,

    'Image Translation Guidelines': `
      <h2>Image Translation Guidelines</h2>
      <p>
        Please adhere to the following principles when evaluating image
        translation:
      </p>
      <ul>
        <li>
          <b>Image is Ground Truth:</b> Always evaluate the translation as
          presented visually in the <b>target image</b> against the original
          <b>source image</b>. Visual cues (logos, icons, spatial arrangement,
          colors, branding, and surrounding graphics) provide essential context
          for whether the translation is accurate and appropriately localized.
        </li>
        <li>
          <b>Transcripts are Only a Selection Tool:</b> The text transcripts
          below each image are provided solely so that you can click on words
          to highlight and record error spans. If the text transcript contains a
          typo, OCR artifact, missing punctuation, or formatting discrepancy
          that is <b>not present in the image</b>, <i>do not mark it as an
          error</i>. Conversely, if an error appears in the image (e.g., a
          misspelled word on a sign, wrong translation, misplaced label, or
          corrupted glyph), mark it by selecting the corresponding word span in
          the transcript.
        </li>
        <li>
          <b>Positioning:</b> Translated text should appear in the
          correct visual location and remain clearly associated with its
          intended visual element. If translated text is placed on the wrong
          button, banner, column, or diagram element in a way that disrupts
          meaning or causes confusion, flag it under <b>Accuracy &gt;
          Positioning</b>. Do not flag minor positioning shifts if the
          visual association remains clear.
        </li>
        <li>
          <b>Text Expansion, Container Fitting, &amp; Clipping:</b> Translated
          text often expands or contracts compared to the source. Look closely
          in the target image for text clipping, container overflow, awkward
          hyphenation across line breaks, overlap collisions with adjacent
          visual elements, or excessive font shrinking that impairs legibility.
        </li>
        <li>
          <b>Character Rendering, Diacritics, &amp; Complex Scripts:</b>
          <ul>
            <li>If characters, accents, diacritics, or script glyphs in the
                image are garbled, corrupted, missing strokes, or unreadable
                (e.g., mojibake or font substitution artifacts), mark them
                under <b>Fluency &gt; Character rendering</b>. (If an accent or
                spelling flaw is an orthographic spelling error on the image
                itself rather than a visual font rendering failure, mark it
                under <b>Fluency &gt; Spelling</b>).</li>
            <li>For Right-to-Left (RTL) and complex scripts, text in the image
                must be properly oriented (reading right-to-left) and cursive
                letters must be appropriately connected rather than
                isolated/disjointed. Flag reversed or disconnected script
                rendering as <b>Fluency &gt; Character rendering</b> (Major
                severity).</li>
          </ul>
        </li>
        <li>
          <b>URLs, Web Addresses, &amp; Brand Names:</b> Hyperlinks, email
          addresses, and domain names in images should generally remain in
          standard web format and not be literally translated (which would break
          the URL). Flag broken or inappropriately translated web addresses
          under <b>Locale convention &gt; URL format</b>. Recognizable brand
          names and trademarks should also generally remain preserved unless an
          official localized brand exists.
        </li>
        <li>
          <b>Click to Expand Images:</b> You can click on any image to expand it
          for a closer, high-resolution inspection. Click again to collapse it
          back to standard size.
        </li>
      </ul>
      `,

    'General Guidelines': `
      <h2>General Guidelines</h2>
      <p>
        The standard you should review the image translations against is
        <b>professional human translation quality</b>. Report every occurrence
        where the translation in the target image falls short of that standard,
        regardless of how the image was produced.
      </p>
      <p>The translation rendered in the target image should be:</p>
      <ul>
        <li>Linguistically correct, accurate, and fluent</li>
        <li>Visually legible, properly positioned, and correctly rendered</li>
        <li>Faithful in tone, register, and style to the source image
            content</li>
        <li>Appropriately localized for the target culture (e.g., currency,
            date, and address conventions)</li>
      </ul>
      <p><b>Important Annotation Principles:</b></p>
      <ol>
        <li>
          <b>Be fine-grained:</b> When multiple independent errors occur in an
          image, log each error separately with its own minimal span.
        </li>
        <li>
          <b>Take visual context into account:</b> A translation choice that
          might look questionable in isolation may be perfectly appropriate
          given the visual context of the image (e.g., concise button labels,
          signs with space constraints).
        </li>
        <li>
          <b>Consistency:</b> If the same error occurs multiple times across an
          image or document, report each instance.
        </li>
        <li>
          <b>Severe / Non-translation cases:</b> If a translated segment in the
          image is completely wrong, unrelated, or pure gibberish, select
          <b>Major severity</b> and use the <b>Non-translation!</b> error type.
        </li>
      </ol>
      `,

    'Navigation': `
      <h2>Navigation</h2>
      <p>
        Each task displays a source image and target image alongside their
        parallel text transcripts:
      </p>
      <ul>
        <li>
          You can navigate through segments using the <b>Left</b> and
          <b>Right Arrow</b> keys, or the arrow buttons in the navigation bar.
        </li>
        <li>
          Click directly on any previously evaluated text segment to jump back
          and review or adjust your annotations.
        </li>
        <li>
          Click on any image to toggle between compact view and expanded
          full-size view.
        </li>
      </ul>
      `,

    'Annotation Process': `
      <h2>Annotation Process</h2>
      <ol>
        <li>
          <b>Inspect the Images:</b> Compare the translation visible in the
          target image against the source image. Identify any translation,
          visual rendering, or layout errors in the target image.
        </li>
        <li>
          <b>Select the Error Span on the Transcript:</b> Click the first word
          where the error begins, then click the word where it ends (for a
          single-word span, click the word twice).
          <ul>
            <li>The text transcript is used as a pointer to identify the
                location of the error in the image.</li>
            <li>For missing content (omissions), select the error span on the
                source transcript side where the omitted content appears.</li>
            <li>Do not mark errors for typos or OCR artifacts that exist only in
                the transcript and not in the image.</li>
          </ul>
        </li>
        <li>
          <b>Select Severity:</b> Choose <b>Major severity (M)</b> or
          <b>Minor severity (m)</b> using the buttons or keyboard shortcuts.
        </li>
        <li>
          <b>Select Category &amp; Subcategory:</b> Pick the appropriate error
          type from the evaluation menu (e.g., <i>Accuracy &gt; Positioning</i>,
          <i>Fluency &gt; Character rendering</i>, <i>Locale
          convention &gt; URL format</i>).
        </li>
        <li>
          <b>Proceed:</b> Use the <b>Right Arrow key</b> to move to the next
          segment after all issues in the current segment have been recorded.
        </li>
      </ol>
      `,

    'Annotation Tips': `
      <details>
        <summary>
          <span class="summary-heading">Annotation Tips</span>
        </summary>
        <ol>
          <li>
            <b>Editing Ratings:</b> You can modify or delete any recorded rating
            in the current segment using the hamburger menu (&#9776;) next to
            the rating.
          </li>
          <li>
            <b>Cancellation:</b> Press <b>Escape</b> or click "Cancel" to abort
            an in-progress error selection.
          </li>
          <li>
            <b>Re-evaluating Previous Segments:</b> Click on any previous
            segment in the document to view, edit, or delete existing
            annotations.
          </li>
          <li>
            <b>Attention Checks:</b> Occasionally, a test segment with an
            artificially injected error may appear to help maintain high
            attention.
          </li>
        </ol>
      </details>
      `,

    'Severities defined': `
      <h2>Severities defined</h2>
      <p>Errors in Visual MQM are classified into two severity levels:</p>
      <ul>
        <li>
          <b>Major severity:</b>
          Errors that significantly alter meaning, mislead the reader, destroy
          comprehension, or severely disrupt visual and linguistic quality in
          the target image.
          <ul>
            <li><i>Examples:</i> Dangerous mistranslations (allergens, safety
                warnings, medical terms), complete omission of key headlines or
                prices, severe text positioning where text is associated with
                the wrong visual element (e.g., swapping "Exit" and "No Exit"
                labels), illegible or corrupted character rendering (mojibake,
                disconnected/reversed Arabic script), or major grammatical
                breakdowns.</li>
          </ul>
        </li>
        <li>
          <b>Minor severity:</b>
          Noticeable flaws in the target image that do not alter the core
          meaning and do not hinder comprehension or disrupt reading flow.
          <ul>
            <li><i>Examples:</i> Minor spelling typos (e.g., "FRESH BRENED
                TEA"), slight text displacement that does not overlap or
                confuse associations, minor punctuation mistakes, minor
                unidiomatic phrasing that remains understandable, or minor
                font/styling mismatches.</li>
          </ul>
        </li>
      </ul>

      <details>
        <summary><b>Severity Classification Examples</b></summary>
        <table>
          <tr>
            <th>Language pair</th>
            <th>Image context</th>
            <th>Source transcript</th>
            <th>Target transcript</th>
            <th>Severity</th>
            <th>Rationale</th>
          </tr>
          <tr>
            <td>EN_FR</td>
            <td>Safety warning sign</td>
            <td>CAUTION: HOT SURFACE</td>
            <td><span class="span-major">ATTENTION: SURFACE FROIDE</span></td>
            <td>Major</td>
            <td>Mistranslated as "cold surface" instead of "hot surface".
                Dangerous safety mistranslation altering meaning
                completely.</td>
          </tr>
          <tr>
            <td>EN_DE</td>
            <td>Coffee shop menu board</td>
            <td>FRESH BREWED COFFEE</td>
            <td>FRISCH <span class="span-minor">GEBRÜHTEN</span> KAFFEE</td>
            <td>Minor</td>
            <td>Minor grammatical ending issue ("gebrühten" instead of
                "gebrühter"), but meaning is completely clear.</td>
          </tr>
          <tr>
            <td>EN_AR</td>
            <td>Storefront banner</td>
            <td>WELCOME</td>
            <td><span class="span-major" dir="ltr"
                >م ر ح ب ا</span></td>
            <td>Major</td>
            <td>Arabic letters rendered left-to-right and disjointed instead
                of connected cursive RTL script. Character rendering major
                error.</td>
          </tr>
          <tr>
            <td>EN_ES</td>
            <td>Promotional flyer footer</td>
            <td>Visit us at www.example.com</td>
            <td>Vis&iacute;tenos en
                <span class="span-major">www.ejemplo.com</span></td>
            <td>Major</td>
            <td>The domain name in the web address was translated, breaking
                the hyperlink URL.</td>
          </tr>
          <tr>
            <td>EN_FR</td>
            <td>Door sign</td>
            <td>PUSH</td>
            <td><span class="span-major">TIREZ</span></td>
            <td>Major</td>
            <td>Positioning/Mistranslation: "Pull" translated on the push
                side of the door sign.</td>
          </tr>
        </table>
      </details>
      `,

    'Error Types and Subtypes defined': `
      <h2>Error Types and Subtypes defined</h2>
      <ul>
        <li>
          <b>Accuracy</b>.
          The translation in the target image does not accurately reflect the
          source content or intent.
          <details open>
            <summary>Subtypes of Accuracy:</summary>
            <ul>
              <li>
                <b>Addition</b>.
                The translated text in the target image includes information or
                content not present in the source image.
                <i>Example:</i> A sign translated with added promotional slogans
                not present on the original image.
              </li>
              <li>
                <b>Mistranslation</b>.
                The translated text in the target image does not accurately
                convey the meaning of the source image.
                <i>Example:</i> A menu item "Roasted Chicken" translated as
                "Boiled Beef".
              </li>
              <li>
                <b>Omission</b>.
                Content present in the source image is missing in the target
                image translation.
                <i>Example:</i> A store sign has opening hours and days, but the
                days of the week are dropped.
                <i>Note:</i> Mark omission errors on the source transcript side.
              </li>
              <li>
                <b>Source language fragment (Untranslated text)</b>.
                Content in the target image that should have been translated has
                been left in the source language.
                <i>Example:</i> A German parking sign has "Parking only for
                residents" left in English.
                <i>Note:</i> Do not flag standard global brand names or
                universal acronyms.
              </li>
              <li>
                <b>Positioning</b>.
                Translated text in the target image is swapped, placed on the
                wrong visual element, or dislocated in a way that degrades
                understanding or visual association.
                <i>Example:</i> On a map or diagram, the label for "Restrooms"
                is placed over the "Cafeteria" icon.
              </li>
            </ul>
          </details>
        </li>

        <li>
          <b>Fluency</b>.
          Issues related to linguistic form, syntax, orthography, or visual
          character rendering in the target image.
          <details open>
            <summary>Subtypes of Fluency:</summary>
            <ul>
              <li>
                <b>Character rendering</b>.
                Visually corrupted, garbled, illegible, or incorrectly rendered
                characters, accents/diacritics, or glyphs in the target image
                (e.g., mojibake, missing font glyphs, or left-to-right reversed
                / disjointed Arabic letters).
              </li>
              <li>
                <b>Grammar</b>.
                Grammatical errors, incorrect word order, or syntactic flaws in
                the target image.
                <i>Example:</i> "They goes here" instead of "They go here".
              </li>
              <li>
                <b>Inconsistency</b>.
                Inconsistent use of tone, honorifics, or naming conventions
                within the translation in the target image.
              </li>
              <li>
                <b>Punctuation</b>.
                Incorrect, missing, or unnecessary punctuation marks or
                quotation styles in the target image.
              </li>
              <li>
                <b>Register</b>.
                Inappropriate grammatical register or level of formality in the
                target image (e.g., using informal "du" instead of formal "Sie"
                on official public signage).
              </li>
              <li>
                <b>Spelling</b>.
                Spelling mistakes, typos, capitalization errors, or
                missing/extra whitespace on the target image itself.
              </li>
            </ul>
          </details>
        </li>

        <li>
          <b>Style</b>.
          Stylistic problems in the target image that affect naturalness and
          readability.
          <details>
            <summary>Subtypes of Style:</summary>
            <ul>
              <li>
                <b>Awkward or unnatural word choice</b>.
                Wording in the target image that is grammatically correct but
                unidiomatic, overly literal, or unnatural in the context of the
                visual element.
              </li>
              <li>
                <b>Bad sentence structure</b>.
                Sentences in the target image structured in an awkward,
                convoluted, or unnecessarily repetitive manner.
              </li>
            </ul>
          </details>
        </li>

        <li>
          <b>Terminology</b>.
          Domain-specific vocabulary in the target image translated
          inappropriately or inconsistently.
          <details>
            <summary>Subtypes of Terminology:</summary>
            <ul>
              <li>
                <b>Inappropriate for context</b>.
                A technical or domain-specific term is translated using a
                generic or incorrect term.
              </li>
              <li>
                <b>Inconsistent use</b>.
                Different terms are used for the same source entity across the
                document/image.
              </li>
            </ul>
          </details>
        </li>

        <li>
          <b>Locale convention</b>.
          Violations of target-locale formatting conventions in the target
          image.
          <details>
            <summary>Subtypes of Locale convention:</summary>
            <ul>
              <li><b>Address format:</b> Postal addresses formatted incorrectly
                  for the target country.</li>
              <li><b>Date format:</b> Inappropriate date order or separators
                  (e.g., <code>DD/MM/YYYY</code> vs
                  <code>MM/DD/YYYY</code>).</li>
              <li><b>Currency format:</b> Incorrect placement or symbol for
                  currency (e.g., <code>100$</code> vs <code>$100</code> or
                  <code>100 &euro;</code>).</li>
              <li><b>Telephone / fax format:</b> Incorrect phone number
                  grouping or country code notation.</li>
              <li><b>Time format:</b> Incorrect time conventions (e.g., 12-hour
                  vs. 24-hour clock).</li>
              <li><b>Name format:</b> Inappropriate personal name ordering.</li>
              <li><b>URL format:</b> Hyperlinks, email addresses, or domain
                  names translated or broken in format.</li>
            </ul>
          </details>
        </li>

        <li>
          <b>Other</b>.
          Any other issue not covered above (a short explanatory note will be
          requested).
        </li>
        <li>
          <b>Non-translation!</b>
          The segment in the target image is completely untranslated gibberish
          or entirely unrelated to the source. Overrides all other annotations
          for the segment.
        </li>
        <li>
          <b>Source issue</b>.
          Any defect or illegibility in the original source image.
        </li>
      </ul>
      `,

    'Annotations exemplified in detail': `
      <details>
        <summary>
          <span class="summary-heading">Detailed Visual MQM Annotation
          Examples</span>
        </summary>
        <table>
          <tr>
            <th>Language pair</th>
            <th>Image context</th>
            <th>Source transcript</th>
            <th>Target transcript</th>
            <th>Annotation</th>
            <th>Comments</th>
          </tr>
          <tr>
            <td>EN_DE</td>
            <td>Road sign</td>
            <td>NO PARKING / TOW AWAY ZONE</td>
            <td><span class="span-major">PARKEN ERLAUBT</span></td>
            <td>Accuracy &gt; Mistranslation (Major)</td>
            <td>Translated as "Parking allowed" instead of "No parking".
                Substantial alteration of safety/regulatory meaning.</td>
          </tr>
          <tr>
            <td>EN_DE</td>
            <td>Emergency safety sign</td>
            <td>In case of emergency, call 911</td>
            <td>Im Notfall <span class="span-minor">112 anrufen</span></td>
            <td>Accuracy &gt; Creative Reinterpretation (Minor)</td>
            <td>The US emergency number (911) is localized to the German/EU
                standard (112) for the target locale.</td>
          </tr>
          <tr>
            <td>EN_ES</td>
            <td>Poster footer</td>
            <td>Visit us at www.citypark.org</td>
            <td>Vis&iacute;tenos en
                <span class="span-major">www.parqueciudad.org</span></td>
            <td>Locale convention &gt; URL format (Major)</td>
            <td>The website domain name was literally translated, resulting in
                a broken web link.</td>
          </tr>
          <tr>
            <td>EN_AR</td>
            <td>Airport sign</td>
            <td>BAGGAGE CLAIM</td>
            <td><span class="span-major" dir="ltr"
                >م ا ل ت س ا</span></td>
            <td>Fluency &gt; Character rendering (Major)</td>
            <td>Arabic letters rendered in reverse direction (LTR) and
                disconnected. Critical character rendering flaw.</td>
          </tr>
          <tr>
            <td>EN_DE</td>
            <td>Museum map exhibit labels</td>
            <td>1. Dinosaurs 2. Minerals</td>
            <td>1. <span class="span-major">Mineralien</span>
                2. <span class="span-major">Dinosaurier</span></td>
            <td>Accuracy &gt; Positioning (Major)</td>
            <td>Labels are swapped on the visual map diagram, associating the
                wrong exhibits with numbers 1 and 2.</td>
          </tr>
          <tr>
            <td>EN_FR</td>
            <td>Restaurant menu item</td>
            <td>Fresh hand-cut fries</td>
            <td>Frites fra&icirc;ches <span
                class="span-minor">coup&eacute;es &agrave; la main</span></td>
            <td>Style &gt; Awkward or unnatural word choice (Minor)</td>
            <td>"Coup&eacute;es &agrave; la main" is grammatically correct but
                unidiomatic on French restaurant menus (better: "frites
                maison").</td>
          </tr>
          <tr>
            <td>EN_DE</td>
            <td>High voltage warning sign</td>
            <td>DANGER: High Voltage</td>
            <td>GEFAHR:
                <span class="span-minor">hochspannung</span></td>
            <td>Fluency &gt; Spelling (Minor)</td>
            <td>"Hochspannung" must be capitalized in German as it is a
                noun.</td>
          </tr>
        </table>
      </details>
      `,
  },

  instructions_section_order: [
    '_style',
    'Overview',
    'Image Translation Guidelines',
    'General Guidelines',
    'Navigation',
    'Annotation Process',
    'Annotation Tips',
    'Severities defined',
    'Error Types and Subtypes defined',
    'Annotations exemplified in detail',
    'Style &amp; Convention Guidelines',
    'Feedback',
  ],
};
